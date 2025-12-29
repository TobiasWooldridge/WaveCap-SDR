import type {
  OverflowIssue,
  FontSizeIssue,
  ViewportClipIssue,
  OverlapIssue,
  ClippedPositionIssue,
  OverflowDetectorConfig,
  ScreenType,
} from './types';
import {
  DEFAULT_EXCLUDED_SELECTORS,
  CLIPPING_OVERFLOW_VALUES,
  SCROLLABLE_OVERFLOW_VALUES,
  FONT_SIZE_HEIGHT_PERCENT,
  FONT_SIZE_MIN_PIXELS,
} from './types';
import { generateSelector, getTextPreview } from './selectors';

// Track reported issues to avoid duplicates
const reportedIssues = new Set<string>();
const reportedFontSizeIssues = new Set<string>();
const reportedViewportClipIssues = new Set<string>();
const reportedOverlapIssues = new Set<string>();
const reportedClippedPositionIssues = new Set<string>();

// Minimum time between reports for the same element (ms)
const REPORT_COOLDOWN = 5000;

// Minimum overflow in pixels to report (avoids sub-pixel rounding issues)
const MIN_VIEWPORT_OVERFLOW = 5;

// Batch throttling
const BATCH_WINDOW_MS = 100;
let overlapBatchTimer: ReturnType<typeof setTimeout> | null = null;
let viewportClipBatchTimer: ReturnType<typeof setTimeout> | null = null;
let overflowBatchTimer: ReturnType<typeof setTimeout> | null = null;

const pendingOverlapIssues: Array<{
  elem1Desc: string;
  elem2Desc: string;
  area: number;
  issue: OverlapIssue;
}> = [];
const pendingViewportClipIssues: Array<{
  elemDesc: string;
  edges: string;
  overflow: number;
  textHint: string;
  issue: ViewportClipIssue;
}> = [];
const pendingOverflowIssues: Array<{
  elemDesc: string;
  direction: string;
  textHint: string;
  issue: OverflowIssue;
}> = [];

/**
 * Detect screen type from viewport dimensions
 */
function detectScreenType(): ScreenType {
  const width = window.innerWidth;
  if (width >= 1200) return 'desktop';
  if (width >= 768) return 'tablet';
  return 'unknown';
}

/**
 * Calculate font size threshold dynamically based on viewport height.
 */
function calculateFontThreshold(screenType: ScreenType): number {
  const referenceHeight = window.innerHeight;
  const percentThreshold = referenceHeight * FONT_SIZE_HEIGHT_PERCENT[screenType];
  return Math.max(percentThreshold, FONT_SIZE_MIN_PIXELS[screenType]);
}

/**
 * OverflowDetector - Detects visual overflow/clipping issues in the DOM
 *
 * Uses ResizeObserver for performance-efficient detection.
 * Only reports elements that have overflow:hidden/clip and are actually clipping content.
 */
export class OverflowDetector {
  private resizeObserver: ResizeObserver | null = null;
  private mutationObserver: MutationObserver | null = null;
  private debounceTimers = new Map<Element, ReturnType<typeof setTimeout>>();
  private config: Required<OverflowDetectorConfig>;
  private logger: { warn: (message: string, data?: unknown) => void };
  private excludedSelectors: string[];
  private isRunning = false;

  constructor(
    logger: { warn: (message: string, data?: unknown) => void },
    config: OverflowDetectorConfig = {}
  ) {
    this.logger = logger;
    this.config = {
      enabled: config.enabled ?? true,
      debounceMs: config.debounceMs ?? 150,
      excludeSelectors: config.excludeSelectors ?? [],
      includeTextPreview: config.includeTextPreview ?? true,
      checkFontSizes: config.checkFontSizes ?? true,
      checkViewportBoundaries: config.checkViewportBoundaries ?? true,
      checkOverlaps: config.checkOverlaps ?? true,
      checkClippedPositions: config.checkClippedPositions ?? true,
      onIssueDetected: config.onIssueDetected ?? (() => {}),
      onFontSizeIssueDetected: config.onFontSizeIssueDetected ?? (() => {}),
      onViewportClipDetected: config.onViewportClipDetected ?? (() => {}),
      onOverlapDetected: config.onOverlapDetected ?? (() => {}),
      onClippedPositionDetected: config.onClippedPositionDetected ?? (() => {}),
    };

    this.excludedSelectors = [...DEFAULT_EXCLUDED_SELECTORS, ...this.config.excludeSelectors];
  }

  /**
   * Start monitoring for overflow issues
   */
  start(): void {
    if (this.isRunning || !this.config.enabled) return;
    this.isRunning = true;

    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        this.scheduleCheck(entry.target as HTMLElement);
      }
    });

    this.mutationObserver = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        mutation.addedNodes.forEach((node) => {
          if (node instanceof HTMLElement) {
            this.observeElement(node);
            node.querySelectorAll('*').forEach((child) => {
              if (child instanceof HTMLElement) {
                this.observeElement(child);
              }
            });
          }
        });
      }
    });

    this.mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });

    this.observeAll();
    setTimeout(() => this.checkAll(), 500);
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (!this.isRunning) return;
    this.isRunning = false;

    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

    this.mutationObserver?.disconnect();
    this.mutationObserver = null;

    this.debounceTimers.forEach((timer) => clearTimeout(timer));
    this.debounceTimers.clear();
  }

  private observeAll(): void {
    document.querySelectorAll('*').forEach((element) => {
      if (element instanceof HTMLElement) {
        this.observeElement(element);
      }
    });
  }

  private observeElement(element: HTMLElement): void {
    if (!this.resizeObserver) return;
    if (this.isExcluded(element)) return;

    try {
      this.resizeObserver.observe(element);
    } catch {
      // Element may have been removed
    }
  }

  private scheduleCheck(element: HTMLElement): void {
    const existingTimer = this.debounceTimers.get(element);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    const timer = setTimeout(() => {
      this.debounceTimers.delete(element);
      this.checkWithIdleCallback(element);
    }, this.config.debounceMs);

    this.debounceTimers.set(element, timer);
  }

  private checkWithIdleCallback(element: HTMLElement): void {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(() => this.checkElement(element), { timeout: 500 });
    } else {
      setTimeout(() => this.checkElement(element), 0);
    }
  }

  private checkAll(): void {
    document.querySelectorAll('*').forEach((element) => {
      if (element instanceof HTMLElement && !this.isExcluded(element)) {
        this.checkElement(element);
      }
    });

    this.checkAllFontSizes();
    this.checkAllViewportBoundaries();
    this.checkAllOverlaps();
    this.checkAllClippedPositions();
  }

  private isExcluded(element: HTMLElement): boolean {
    for (const selector of this.excludedSelectors) {
      try {
        if (element.matches(selector)) {
          return true;
        }
      } catch {
        // Invalid selector, skip
      }
    }

    const style = getComputedStyle(element);
    if (
      SCROLLABLE_OVERFLOW_VALUES.includes(style.overflow) ||
      SCROLLABLE_OVERFLOW_VALUES.includes(style.overflowX) ||
      SCROLLABLE_OVERFLOW_VALUES.includes(style.overflowY)
    ) {
      return true;
    }

    return false;
  }

  private checkElement(element: HTMLElement): void {
    if (!element.isConnected) return;
    if (this.isExcluded(element)) return;

    const hasHorizontalOverflow = element.scrollWidth > element.clientWidth;
    const hasVerticalOverflow = element.scrollHeight > element.clientHeight;

    if (!hasHorizontalOverflow && !hasVerticalOverflow) return;

    const style = getComputedStyle(element);
    const isClippingX = CLIPPING_OVERFLOW_VALUES.includes(style.overflowX);
    const isClippingY = CLIPPING_OVERFLOW_VALUES.includes(style.overflowY);
    const isClipping = CLIPPING_OVERFLOW_VALUES.includes(style.overflow);

    const isHorizontallyClipped = hasHorizontalOverflow && (isClippingX || isClipping);
    const isVerticallyClipped = hasVerticalOverflow && (isClippingY || isClipping);

    if (!isHorizontallyClipped && !isVerticallyClipped) return;

    const issue = this.createIssue(element, {
      horizontal: isHorizontallyClipped,
      vertical: isVerticallyClipped,
    });

    this.reportIssue(issue);
  }

  private createIssue(
    element: HTMLElement,
    overflow: { horizontal: boolean; vertical: boolean }
  ): OverflowIssue {
    const style = getComputedStyle(element);

    return {
      selector: generateSelector(element),
      elementTag: element.tagName.toLowerCase(),
      overflow,
      dimensions: {
        scrollWidth: element.scrollWidth,
        scrollHeight: element.scrollHeight,
        clientWidth: element.clientWidth,
        clientHeight: element.clientHeight,
      },
      computedStyles: {
        overflow: style.overflow,
        overflowX: style.overflowX,
        overflowY: style.overflowY,
        textOverflow: style.textOverflow,
      },
      textPreview: this.config.includeTextPreview ? getTextPreview(element) : undefined,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      timestamp: Date.now(),
    };
  }

  private reportIssue(issue: OverflowIssue): void {
    const issueKey = `${issue.selector}:${issue.overflow.horizontal}:${issue.overflow.vertical}`;

    if (reportedIssues.has(issueKey)) {
      return;
    }

    reportedIssues.add(issueKey);

    setTimeout(() => {
      reportedIssues.delete(issueKey);
    }, REPORT_COOLDOWN);

    const elemDesc = this.formatElementDescription(issue.selector, issue.elementTag);
    const direction =
      issue.overflow.horizontal && issue.overflow.vertical
        ? 'horizontally and vertically'
        : issue.overflow.horizontal
          ? 'horizontally'
          : 'vertically';
    const textHint = issue.textPreview ? ` (text: "${issue.textPreview.slice(0, 30)}...")` : '';

    pendingOverflowIssues.push({
      elemDesc,
      direction,
      textHint,
      issue,
    });

    this.config.onIssueDetected(issue);

    if (!overflowBatchTimer) {
      overflowBatchTimer = setTimeout(() => {
        this.flushOverflowBatch();
      }, BATCH_WINDOW_MS);
    }
  }

  private flushOverflowBatch(): void {
    overflowBatchTimer = null;

    if (pendingOverflowIssues.length === 0) return;

    const viewportStr = `${window.innerWidth}x${window.innerHeight}`;
    const first = pendingOverflowIssues[0];
    if (pendingOverflowIssues.length === 1 && first) {
      const { elemDesc, direction, textHint, issue } = first;
      this.logger.warn(
        `Visual overflow detected @ ${viewportStr}: ${elemDesc} overflows ${direction}${textHint}`,
        {
          category: 'visual-overflow',
          issue,
        }
      );
    } else {
      const count = pendingOverflowIssues.length;
      const details = pendingOverflowIssues
        .slice(0, 5)
        .map(({ elemDesc, direction }) => `${elemDesc} (${direction})`)
        .join(', ');
      const suffix = count > 5 ? `, +${count - 5} more` : '';

      this.logger.warn(
        `Visual overflow detected @ ${viewportStr} (${count} elements): ${details}${suffix}`,
        {
          category: 'visual-overflow',
          issues: pendingOverflowIssues.map((p) => p.issue),
        }
      );
    }

    pendingOverflowIssues.length = 0;
  }

  private checkAllFontSizes(): void {
    if (!this.config.checkFontSizes) return;

    if (window.innerHeight < 600) return;

    const screenType = detectScreenType();
    const threshold = calculateFontThreshold(screenType);

    const textElements = document.querySelectorAll(
      'p, span, div, h1, h2, h3, h4, h5, h6, button, a, label, li, td, th, small'
    );

    textElements.forEach((element) => {
      if (element instanceof HTMLElement) {
        this.checkElementFontSize(element, screenType, threshold);
      }
    });
  }

  private checkElementFontSize(
    element: HTMLElement,
    screenType: ScreenType,
    threshold: number
  ): void {
    const hasDirectText = Array.from(element.childNodes).some(
      (node) => node.nodeType === Node.TEXT_NODE && node.textContent?.trim()
    );
    if (!hasDirectText) return;

    if (this.isExcluded(element)) return;

    const style = getComputedStyle(element);
    const fontSize = parseFloat(style.fontSize);

    if (fontSize >= threshold) return;

    const issue: FontSizeIssue = {
      selector: generateSelector(element),
      elementTag: element.tagName.toLowerCase(),
      fontSize: Math.round(fontSize * 10) / 10,
      minimumFontSize: threshold,
      screenType,
      textPreview: this.config.includeTextPreview ? getTextPreview(element) : undefined,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      timestamp: Date.now(),
    };

    this.reportFontSizeIssue(issue);
  }

  private reportFontSizeIssue(issue: FontSizeIssue): void {
    const issueKey = `${issue.selector}:${issue.fontSize}`;

    if (reportedFontSizeIssues.has(issueKey)) {
      return;
    }

    reportedFontSizeIssues.add(issueKey);

    setTimeout(() => {
      reportedFontSizeIssues.delete(issueKey);
    }, REPORT_COOLDOWN);

    const textHint = issue.textPreview ? ` (text: "${issue.textPreview.slice(0, 30)}...")` : '';
    this.logger.warn(
      `Font size too small for screen type: ${issue.fontSize}px < ${issue.minimumFontSize}px min @ ${issue.viewport.width}x${issue.viewport.height}${textHint}`,
      {
        category: 'font-size',
        issue,
      }
    );

    this.config.onFontSizeIssueDetected(issue);
  }

  private checkAllViewportBoundaries(): void {
    if (!this.config.checkViewportBoundaries) return;

    const elementsToCheck = document.querySelectorAll(
      'p, span, div, h1, h2, h3, h4, h5, h6, button, a, label, img, svg, input, select, textarea'
    );

    elementsToCheck.forEach((element) => {
      if (element instanceof HTMLElement) {
        this.checkElementViewportClip(element);
      }
    });
  }

  private checkElementViewportClip(element: HTMLElement): void {
    if (!element.isConnected) return;
    if (this.isExcluded(element)) return;

    const style = getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
      return;
    }

    const rect = element.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const vw = window.innerWidth;
    const vh = window.innerHeight;

    const clipping = {
      top: rect.top < 0,
      bottom: rect.bottom > vh,
      left: rect.left < 0,
      right: rect.right > vw,
    };

    if (!clipping.top && !clipping.bottom && !clipping.left && !clipping.right) {
      return;
    }

    const overflow = Math.max(
      clipping.top ? -rect.top : 0,
      clipping.bottom ? rect.bottom - vh : 0,
      clipping.left ? -rect.left : 0,
      clipping.right ? rect.right - vw : 0
    );

    if (overflow < MIN_VIEWPORT_OVERFLOW) return;

    const issue: ViewportClipIssue = {
      selector: generateSelector(element),
      elementTag: element.tagName.toLowerCase(),
      clipping,
      bounds: {
        top: Math.round(rect.top),
        right: Math.round(rect.right),
        bottom: Math.round(rect.bottom),
        left: Math.round(rect.left),
      },
      viewport: {
        width: vw,
        height: vh,
      },
      overflow: Math.round(overflow),
      textPreview: this.config.includeTextPreview ? getTextPreview(element) : undefined,
      timestamp: Date.now(),
    };

    this.reportViewportClipIssue(issue);
  }

  private reportViewportClipIssue(issue: ViewportClipIssue): void {
    const clipDir = [
      issue.clipping.top ? 'T' : '',
      issue.clipping.bottom ? 'B' : '',
      issue.clipping.left ? 'L' : '',
      issue.clipping.right ? 'R' : '',
    ].join('');
    const issueKey = `viewport:${issue.selector}:${clipDir}`;

    if (reportedViewportClipIssues.has(issueKey)) {
      return;
    }

    reportedViewportClipIssues.add(issueKey);

    setTimeout(() => {
      reportedViewportClipIssues.delete(issueKey);
    }, REPORT_COOLDOWN);

    const elemDesc = this.formatElementDescription(issue.selector, issue.elementTag);
    const edges = [
      issue.clipping.top ? 'top' : '',
      issue.clipping.bottom ? 'bottom' : '',
      issue.clipping.left ? 'left' : '',
      issue.clipping.right ? 'right' : '',
    ]
      .filter(Boolean)
      .join(', ');
    const textHint = issue.textPreview ? ` (text: "${issue.textPreview.slice(0, 30)}...")` : '';

    pendingViewportClipIssues.push({
      elemDesc,
      edges,
      overflow: issue.overflow,
      textHint,
      issue,
    });

    this.config.onViewportClipDetected(issue);

    if (!viewportClipBatchTimer) {
      viewportClipBatchTimer = setTimeout(() => {
        this.flushViewportClipBatch();
      }, BATCH_WINDOW_MS);
    }
  }

  private flushViewportClipBatch(): void {
    viewportClipBatchTimer = null;

    if (pendingViewportClipIssues.length === 0) return;

    const viewportStr = `${window.innerWidth}x${window.innerHeight}`;
    const first = pendingViewportClipIssues[0];
    if (pendingViewportClipIssues.length === 1 && first) {
      const { elemDesc, edges, overflow, textHint, issue } = first;
      this.logger.warn(
        `Viewport boundary clipping @ ${viewportStr}: ${elemDesc} extends ${overflow}px beyond ${edges}${textHint}`,
        {
          category: 'viewport-clip',
          issue,
        }
      );
    } else {
      const count = pendingViewportClipIssues.length;
      const details = pendingViewportClipIssues
        .slice(0, 5)
        .map(({ elemDesc, edges, overflow }) => `${elemDesc} (${overflow}px beyond ${edges})`)
        .join(', ');
      const suffix = count > 5 ? `, +${count - 5} more` : '';

      this.logger.warn(
        `Viewport boundary clipping @ ${viewportStr} (${count} elements): ${details}${suffix}`,
        {
          category: 'viewport-clip',
          issues: pendingViewportClipIssues.map((p) => p.issue),
        }
      );
    }

    pendingViewportClipIssues.length = 0;
  }

  private checkAllOverlaps(): void {
    if (!this.config.checkOverlaps) return;

    const containers = document.querySelectorAll(
      '.d-flex, .d-grid, .row, .col, .position-relative, [class*="flex"], [class*="grid"]'
    );

    containers.forEach((container) => {
      if (container instanceof HTMLElement) {
        this.checkContainerOverlaps(container);
      }
    });
  }

  private checkContainerOverlaps(container: HTMLElement): void {
    if (!container.isConnected) return;
    if (this.isExcluded(container)) return;

    const children = Array.from(container.children).filter((child): child is HTMLElement => {
      if (!(child instanceof HTMLElement)) return false;
      const style = getComputedStyle(child);
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
        return false;
      }
      return true;
    });

    if (children.length < 2) return;

    for (let i = 0; i < children.length; i++) {
      for (let j = i + 1; j < children.length; j++) {
        const child1 = children[i];
        const child2 = children[j];
        if (!child1 || !child2) continue;

        const rect1 = child1.getBoundingClientRect();
        const rect2 = child2.getBoundingClientRect();

        if (rect1.width === 0 || rect1.height === 0) continue;
        if (rect2.width === 0 || rect2.height === 0) continue;

        const overlapX = Math.max(
          0,
          Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left)
        );
        const overlapY = Math.max(
          0,
          Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top)
        );
        const overlapArea = overlapX * overlapY;

        if (overlapArea > 4) {
          const issue: OverlapIssue = {
            parentSelector: generateSelector(container),
            elements: [
              {
                selector: generateSelector(child1),
                tag: child1.tagName.toLowerCase(),
                bounds: {
                  top: Math.round(rect1.top),
                  left: Math.round(rect1.left),
                  width: Math.round(rect1.width),
                  height: Math.round(rect1.height),
                },
              },
              {
                selector: generateSelector(child2),
                tag: child2.tagName.toLowerCase(),
                bounds: {
                  top: Math.round(rect2.top),
                  left: Math.round(rect2.left),
                  width: Math.round(rect2.width),
                  height: Math.round(rect2.height),
                },
              },
            ],
            overlapArea: Math.round(overlapArea),
            timestamp: Date.now(),
          };

          this.reportOverlapIssue(issue);
        }
      }
    }
  }

  private reportOverlapIssue(issue: OverlapIssue): void {
    const issueKey = `overlap:${issue.elements[0].selector}:${issue.elements[1].selector}`;

    if (reportedOverlapIssues.has(issueKey)) {
      return;
    }

    reportedOverlapIssues.add(issueKey);

    setTimeout(() => {
      reportedOverlapIssues.delete(issueKey);
    }, REPORT_COOLDOWN);

    const elem1 = issue.elements[0];
    const elem2 = issue.elements[1];
    const elem1Desc = this.formatElementDescription(elem1.selector, elem1.tag);
    const elem2Desc = this.formatElementDescription(elem2.selector, elem2.tag);

    pendingOverlapIssues.push({
      elem1Desc,
      elem2Desc,
      area: issue.overlapArea,
      issue,
    });

    this.config.onOverlapDetected(issue);

    if (!overlapBatchTimer) {
      overlapBatchTimer = setTimeout(() => {
        this.flushOverlapBatch();
      }, BATCH_WINDOW_MS);
    }
  }

  private flushOverlapBatch(): void {
    overlapBatchTimer = null;

    if (pendingOverlapIssues.length === 0) return;

    const viewportStr = `${window.innerWidth}x${window.innerHeight}`;
    const first = pendingOverlapIssues[0];
    if (pendingOverlapIssues.length === 1 && first) {
      const { elem1Desc, elem2Desc, area, issue } = first;
      this.logger.warn(
        `Sibling element overlap @ ${viewportStr}: ${elem1Desc} overlaps ${elem2Desc} by ${area}px^2`,
        {
          category: 'overlap',
          issue,
        }
      );
    } else {
      const count = pendingOverlapIssues.length;
      const details = pendingOverlapIssues
        .slice(0, 5)
        .map(({ elem1Desc, elem2Desc, area }) => `${elem1Desc}<->${elem2Desc} (${area}px^2)`)
        .join(', ');
      const suffix = count > 5 ? `, +${count - 5} more` : '';

      this.logger.warn(
        `Sibling element overlap @ ${viewportStr} (${count} pairs): ${details}${suffix}`,
        {
          category: 'overlap',
          issues: pendingOverlapIssues.map((p) => p.issue),
        }
      );
    }

    pendingOverlapIssues.length = 0;
  }

  private checkAllClippedPositions(): void {
    if (!this.config.checkClippedPositions) return;

    const positionedElements = document.querySelectorAll('*');

    positionedElements.forEach((element) => {
      if (element instanceof HTMLElement) {
        this.checkElementClippedByAncestor(element);
      }
    });
  }

  private checkElementClippedByAncestor(element: HTMLElement): void {
    if (!element.isConnected) return;
    if (this.isExcluded(element)) return;

    const style = getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
      return;
    }

    const position = style.position;
    if (position !== 'absolute' && position !== 'fixed') return;

    const elementRect = element.getBoundingClientRect();
    if (elementRect.width === 0 || elementRect.height === 0) return;

    let ancestor = element.parentElement;
    while (ancestor) {
      const ancestorStyle = getComputedStyle(ancestor);

      const clipsOverflow =
        CLIPPING_OVERFLOW_VALUES.includes(ancestorStyle.overflow) ||
        CLIPPING_OVERFLOW_VALUES.includes(ancestorStyle.overflowX) ||
        CLIPPING_OVERFLOW_VALUES.includes(ancestorStyle.overflowY);

      if (clipsOverflow) {
        const ancestorRect = ancestor.getBoundingClientRect();

        const overflowTop = ancestorRect.top - elementRect.top;
        const overflowBottom = elementRect.bottom - ancestorRect.bottom;
        const overflowLeft = ancestorRect.left - elementRect.left;
        const overflowRight = elementRect.right - ancestorRect.right;

        const isClipped =
          overflowTop > 1 || overflowBottom > 1 || overflowLeft > 1 || overflowRight > 1;

        if (isClipped) {
          const issue: ClippedPositionIssue = {
            selector: generateSelector(element),
            elementTag: element.tagName.toLowerCase(),
            ancestorSelector: generateSelector(ancestor),
            clipping: {
              top: overflowTop > 1,
              bottom: overflowBottom > 1,
              left: overflowLeft > 1,
              right: overflowRight > 1,
            },
            overflow: {
              top: Math.max(0, Math.round(overflowTop)),
              bottom: Math.max(0, Math.round(overflowBottom)),
              left: Math.max(0, Math.round(overflowLeft)),
              right: Math.max(0, Math.round(overflowRight)),
            },
            textPreview: this.config.includeTextPreview ? getTextPreview(element) : undefined,
            timestamp: Date.now(),
          };

          this.reportClippedPositionIssue(issue);
          return;
        }
      }

      ancestor = ancestor.parentElement;
    }
  }

  private reportClippedPositionIssue(issue: ClippedPositionIssue): void {
    const clipDir = [
      issue.clipping.top ? 'T' : '',
      issue.clipping.bottom ? 'B' : '',
      issue.clipping.left ? 'L' : '',
      issue.clipping.right ? 'R' : '',
    ].join('');
    const issueKey = `clipped:${issue.selector}:${issue.ancestorSelector}:${clipDir}`;

    if (reportedClippedPositionIssues.has(issueKey)) {
      return;
    }

    reportedClippedPositionIssues.add(issueKey);

    setTimeout(() => {
      reportedClippedPositionIssues.delete(issueKey);
    }, REPORT_COOLDOWN);

    const elemDesc = this.formatElementDescription(issue.selector, issue.elementTag);
    const ancestorDesc = this.formatElementDescription(issue.ancestorSelector, 'div');
    const edges = [
      issue.clipping.top ? `top:${issue.overflow.top}px` : '',
      issue.clipping.bottom ? `bottom:${issue.overflow.bottom}px` : '',
      issue.clipping.left ? `left:${issue.overflow.left}px` : '',
      issue.clipping.right ? `right:${issue.overflow.right}px` : '',
    ]
      .filter(Boolean)
      .join(', ');
    const textHint = issue.textPreview ? ` (content: "${issue.textPreview.slice(0, 20)}...")` : '';

    const viewportStr = `${window.innerWidth}x${window.innerHeight}`;
    this.logger.warn(
      `Positioned element clipped @ ${viewportStr}: ${elemDesc} extends beyond ${ancestorDesc} (${edges})${textHint}`,
      {
        category: 'clipped-position',
        issue,
      }
    );

    this.config.onClippedPositionDetected(issue);
  }

  private formatElementDescription(selector: string, tag: string): string {
    const testIdMatch = selector.match(/\[data-testid="([^"]+)"\]/);
    if (testIdMatch) {
      return `[${testIdMatch[1]}]`;
    }

    if (selector.startsWith('#')) {
      return selector;
    }

    const classMatch = selector.match(/^([a-z]+)\.([^\s>]+)/i);
    if (classMatch && classMatch[1] && classMatch[2]) {
      const classes = classMatch[2].split('.').slice(0, 2).join('.');
      return `<${classMatch[1]}.${classes}>`;
    }

    return `<${tag}>`;
  }
}
