import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ComponentPropsWithRef,
  type ElementType,
  type ForwardedRef,
  type ReactElement,
} from "react";
import clsx from "clsx";

const BREAKPOINTS = ["sm", "md", "lg", "xl", "xxl"] as const;

export type FlexBreakpoint = (typeof BREAKPOINTS)[number];

type ResponsiveMap<Value> = {
  base?: Value;
} & Partial<Record<FlexBreakpoint, Value>>;

export type ResponsiveProp<Value> = Value | ResponsiveMap<Value>;

type FlexDirection = "row" | "row-reverse" | "column" | "column-reverse";
type FlexWrap = "wrap" | "nowrap" | "wrap-reverse";
type FlexJustify = "start" | "end" | "center" | "between" | "around" | "evenly";
type FlexAlign = "start" | "end" | "center" | "baseline" | "stretch";
type FlexAlignContent = "start" | "end" | "center" | "between" | "around" | "stretch";
type FlexGap = 0 | 1 | 2 | 3 | 4 | 5;

type PolymorphicRef<C extends ElementType> = ComponentPropsWithRef<C>["ref"];

type FlexOwnProps<C extends ElementType> = {
  as?: C;
  inline?: boolean;
  direction?: ResponsiveProp<FlexDirection>;
  wrap?: ResponsiveProp<FlexWrap>;
  justify?: ResponsiveProp<FlexJustify>;
  align?: ResponsiveProp<FlexAlign>;
  alignContent?: ResponsiveProp<FlexAlignContent>;
  gap?: ResponsiveProp<FlexGap>;
} & {
  className?: string;
};

type FlexProps<C extends ElementType> = FlexOwnProps<C> &
  Omit<ComponentPropsWithoutRef<C>, keyof FlexOwnProps<C> | "as">;

type FlexComponent = (<C extends ElementType = "div">(
  props: FlexProps<C> & { ref?: PolymorphicRef<C> },
) => ReactElement | null) & { displayName?: string };

const toArray = (value: string | string[] | undefined): string[] => {
  if (typeof value === "string") {
    return [value];
  }

  return Array.isArray(value) ? value : [];
};

const isResponsiveMap = <Value,>(value: ResponsiveProp<Value>): value is ResponsiveMap<Value> => {
  return typeof value === "object" && value !== null;
};

const resolveResponsiveProp = <Value,>(
  propValue: ResponsiveProp<Value> | undefined,
  generator: (value: Value, breakpoint?: FlexBreakpoint) => string | string[] | undefined,
): string[] => {
  if (propValue == null) {
    return [];
  }

  if (!isResponsiveMap(propValue)) {
    return toArray(generator(propValue));
  }

  const classNames: string[] = [];

  if (propValue.base != null) {
    classNames.push(...toArray(generator(propValue.base)));
  }

  for (const breakpoint of BREAKPOINTS) {
    const value = propValue[breakpoint];
    if (value != null) {
      classNames.push(...toArray(generator(value, breakpoint)));
    }
  }

  return classNames;
};

const resolveDirectionClass = (direction: FlexDirection, breakpoint?: FlexBreakpoint) => {
  const prefix = breakpoint ? `flex-${breakpoint}` : "flex";

  switch (direction) {
    case "row":
      return `${prefix}-row`;
    case "row-reverse":
      return `${prefix}-row-reverse`;
    case "column":
      return `${prefix}-column`;
    case "column-reverse":
      return `${prefix}-column-reverse`;
    default:
      return undefined;
  }
};

const resolveWrapClass = (wrap: FlexWrap, breakpoint?: FlexBreakpoint) => {
  const prefix = breakpoint ? `flex-${breakpoint}` : "flex";

  switch (wrap) {
    case "wrap":
      return `${prefix}-wrap`;
    case "wrap-reverse":
      return `${prefix}-wrap-reverse`;
    case "nowrap":
      return `${prefix}-nowrap`;
    default:
      return undefined;
  }
};

const resolveJustifyClass = (justify: FlexJustify, breakpoint?: FlexBreakpoint) => {
  const prefix = breakpoint ? `justify-content-${breakpoint}` : "justify-content";
  return `${prefix}-${justify}`;
};

const resolveAlignItemsClass = (align: FlexAlign, breakpoint?: FlexBreakpoint) => {
  const prefix = breakpoint ? `align-items-${breakpoint}` : "align-items";
  return `${prefix}-${align}`;
};

const resolveAlignContentClass = (align: FlexAlignContent, breakpoint?: FlexBreakpoint) => {
  const prefix = breakpoint ? `align-content-${breakpoint}` : "align-content";
  return `${prefix}-${align}`;
};

const resolveGapClass = (gap: FlexGap, breakpoint?: FlexBreakpoint) => {
  const prefix = breakpoint ? `gap-${breakpoint}` : "gap";
  return `${prefix}-${gap}`;
};

const Flex = forwardRef(
  <C extends ElementType = "div">(
    {
      as,
      inline = false,
      direction,
      wrap,
      justify,
      align,
      alignContent,
      gap,
      className,
      ...rest
    }: FlexProps<C>,
    forwardedRef: ForwardedRef<unknown>,
  ) => {
    const ref = forwardedRef as PolymorphicRef<C>;
    const Component = (as ?? "div") as ElementType;

    const directionClasses = resolveResponsiveProp(direction, resolveDirectionClass);
    const wrapClasses = resolveResponsiveProp(wrap, resolveWrapClass);
    const justifyClasses = resolveResponsiveProp(justify, resolveJustifyClass);
    const alignClasses = resolveResponsiveProp(align, resolveAlignItemsClass);
    const alignContentClasses = resolveResponsiveProp(alignContent, resolveAlignContentClass);
    const gapClasses = resolveResponsiveProp(gap, resolveGapClass);

    return (
      <Component
        {...rest}
        ref={ref}
        className={clsx(
          inline ? "d-inline-flex" : "d-flex",
          directionClasses,
          wrapClasses,
          justifyClasses,
          alignClasses,
          alignContentClasses,
          gapClasses,
          className,
        )}
      />
    );
  },
) as FlexComponent;

Flex.displayName = "Flex";

export type { FlexProps };

export default Flex;
