/**
 * Screen type for context-appropriate thresholds
 * WaveCap-SDR is primarily a desktop app, but we support tablets too
 */
export type ScreenType = 'desktop' | 'tablet' | 'unknown';

/**
 * Font size threshold as percentage of viewport height.
 * Desktop apps need readable fonts for long monitoring sessions.
 */
export const FONT_SIZE_HEIGHT_PERCENT: Record<ScreenType, number> = {
  desktop: 0.012, // 1.2% of height (~13px on 1080p)
  tablet: 0.014,  // 1.4% for tablet readability
  unknown: 0.012,
};

/**
 * Absolute minimum font sizes (in pixels) regardless of viewport size.
 */
export const FONT_SIZE_MIN_PIXELS: Record<ScreenType, number> = {
  desktop: 11, // 11px minimum for dense data displays
  tablet: 12,  // 12px for tablet touch targets
  unknown: 11,
};

/**
 * Information about a detected font size issue
 */
export interface FontSizeIssue {
  /** CSS selector path to the element */
  selector: string;
  /** HTML tag name of the element */
  elementTag: string;
  /** Actual font size in pixels */
  fontSize: number;
  /** Minimum expected font size for this screen */
  minimumFontSize: number;
  /** Screen type that was detected */
  screenType: ScreenType;
  /** First 50 chars of text content for context */
  textPreview?: string;
  /** Viewport dimensions when issue was detected */
  viewport: {
    width: number;
    height: number;
  };
  /** Timestamp of detection */
  timestamp: number;
}

/**
 * Information about a detected viewport boundary clipping issue
 */
export interface ViewportClipIssue {
  /** CSS selector path to the element */
  selector: string;
  /** HTML tag name of the element */
  elementTag: string;
  /** Which edges are clipped */
  clipping: {
    top: boolean;
    bottom: boolean;
    left: boolean;
    right: boolean;
  };
  /** Element bounding rect */
  bounds: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  /** Viewport dimensions */
  viewport: {
    width: number;
    height: number;
  };
  /** Maximum overflow in pixels */
  overflow: number;
  /** First 50 chars of text content for context */
  textPreview?: string;
  /** Timestamp of detection */
  timestamp: number;
}

/**
 * Information about a positioned element being clipped by an overflow-hidden ancestor
 */
export interface ClippedPositionIssue {
  /** CSS selector path to the clipped element */
  selector: string;
  /** HTML tag name of the element */
  elementTag: string;
  /** CSS selector path to the overflow-hidden ancestor */
  ancestorSelector: string;
  /** Which edges are being clipped */
  clipping: {
    top: boolean;
    bottom: boolean;
    left: boolean;
    right: boolean;
  };
  /** How much the element extends beyond the ancestor bounds (in pixels) */
  overflow: {
    top: number;
    bottom: number;
    left: number;
    right: number;
  };
  /** First 50 chars of text content for context */
  textPreview?: string;
  /** Timestamp of detection */
  timestamp: number;
}

/**
 * Information about a detected element overlap issue
 */
export interface OverlapIssue {
  /** CSS selector path to the parent container */
  parentSelector: string;
  /** The two overlapping elements */
  elements: [
    {
      selector: string;
      tag: string;
      bounds: { top: number; left: number; width: number; height: number };
    },
    {
      selector: string;
      tag: string;
      bounds: { top: number; left: number; width: number; height: number };
    },
  ];
  /** Area of overlap in pixels squared */
  overlapArea: number;
  /** Timestamp of detection */
  timestamp: number;
}

/**
 * Information about a detected visual overflow issue
 */
export interface OverflowIssue {
  /** CSS selector path to the element */
  selector: string;
  /** HTML tag name of the element */
  elementTag: string;
  /** Which directions have overflow */
  overflow: {
    horizontal: boolean;
    vertical: boolean;
  };
  /** Dimension measurements */
  dimensions: {
    scrollWidth: number;
    scrollHeight: number;
    clientWidth: number;
    clientHeight: number;
  };
  /** Relevant computed styles */
  computedStyles: {
    overflow: string;
    overflowX: string;
    overflowY: string;
    textOverflow: string;
  };
  /** First 50 chars of text content for context */
  textPreview?: string;
  /** Viewport dimensions when issue was detected */
  viewport: {
    width: number;
    height: number;
  };
  /** Timestamp of detection */
  timestamp: number;
}

/**
 * Configuration options for the overflow detector
 */
export interface OverflowDetectorConfig {
  /** Enable detection (default: true in dev, false in prod) */
  enabled?: boolean;
  /** Debounce time in ms for resize events (default: 150) */
  debounceMs?: number;
  /** Additional selectors to exclude from checking */
  excludeSelectors?: string[];
  /** Include text preview in logs (default: true) */
  includeTextPreview?: boolean;
  /** Enable font size checking (default: true) */
  checkFontSizes?: boolean;
  /** Enable viewport boundary checking (default: true) */
  checkViewportBoundaries?: boolean;
  /** Enable sibling overlap checking (default: true) */
  checkOverlaps?: boolean;
  /** Enable detection of elements clipped by overflow-hidden ancestors (default: true) */
  checkClippedPositions?: boolean;
  /** Callback when an overflow issue is detected */
  onIssueDetected?: (issue: OverflowIssue) => void;
  /** Callback when a font size issue is detected */
  onFontSizeIssueDetected?: (issue: FontSizeIssue) => void;
  /** Callback when viewport boundary clipping is detected */
  onViewportClipDetected?: (issue: ViewportClipIssue) => void;
  /** Callback when sibling element overlap is detected */
  onOverlapDetected?: (issue: OverlapIssue) => void;
  /** Callback when a positioned element is clipped by overflow-hidden ancestor */
  onClippedPositionDetected?: (issue: ClippedPositionIssue) => void;
}

/**
 * Default selectors to exclude from overflow checking
 * These are intentionally scrollable or truncated elements
 */
export const DEFAULT_EXCLUDED_SELECTORS = [
  // Intentionally scrollable containers
  '[data-scroll-container]',
  '.overflow-y-auto',
  '.overflow-x-auto',
  '.overflow-auto',
  '.overflow-scroll',

  // Intentional truncation
  '.text-truncate',
  '.text-nowrap',

  // Animation containers (transient states)
  '[class*="animate-"]',
  '.transitioning',

  // Dev/debug elements
  '[class*="DevOverlay"]',
  '[data-dev-only]',

  // Spectrum/waterfall canvases (expected to have internal scrolling)
  'canvas',
  '.spectrum-container',
  '.waterfall-container',

  // Bootstrap components with intentional overflow
  '.dropdown-menu',
  '.modal',
  '.modal-dialog',
  '.offcanvas',
  '.tooltip',
  '.popover',

  // Intentionally decorative/overlapping elements
  '[data-decorative]',
];

/**
 * Overflow values that indicate intentional scrolling
 */
export const SCROLLABLE_OVERFLOW_VALUES = ['scroll', 'auto'];

/**
 * Overflow values that cause clipping
 */
export const CLIPPING_OVERFLOW_VALUES = ['hidden', 'clip'];
