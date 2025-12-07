import { useState, createContext, useContext, type ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

// Context for accordion group (allows only one open at a time if needed)
interface AccordionContextValue {
  openId: string | null;
  setOpenId: (id: string | null) => void;
  allowMultiple: boolean;
}

const AccordionContext = createContext<AccordionContextValue | null>(null);

interface AccordionGroupProps {
  children: ReactNode;
  allowMultiple?: boolean;
  defaultOpen?: string;
}

export function AccordionGroup({ children, allowMultiple = false, defaultOpen }: AccordionGroupProps) {
  const [openId, setOpenId] = useState<string | null>(defaultOpen ?? null);

  return (
    <AccordionContext.Provider value={{ openId, setOpenId, allowMultiple }}>
      <div className="d-flex flex-column gap-1">{children}</div>
    </AccordionContext.Provider>
  );
}

interface AccordionItemProps {
  id: string;
  header: ReactNode;
  headerRight?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}

export function AccordionItem({
  id,
  header,
  headerRight,
  children,
  defaultOpen = false,
  className = "",
}: AccordionItemProps) {
  const context = useContext(AccordionContext);
  const [localOpen, setLocalOpen] = useState(defaultOpen);

  // Use context if available, otherwise use local state
  const isOpen = context ? (context.allowMultiple ? localOpen : context.openId === id) : localOpen;

  const toggle = () => {
    if (context) {
      if (context.allowMultiple) {
        setLocalOpen(!localOpen);
      } else {
        context.setOpenId(isOpen ? null : id);
      }
    } else {
      setLocalOpen(!localOpen);
    }
  };

  return (
    <div className={`accordion-item border rounded ${className}`}>
      <div
        className="accordion-header d-flex align-items-center justify-content-between px-2 py-1 bg-body-tertiary"
        style={{ cursor: "pointer", minHeight: "32px" }}
        onClick={toggle}
      >
        <div className="d-flex align-items-center gap-1 flex-grow-1 overflow-hidden">
          {isOpen ? (
            <ChevronDown size={14} className="flex-shrink-0 text-muted" />
          ) : (
            <ChevronRight size={14} className="flex-shrink-0 text-muted" />
          )}
          <div className="overflow-hidden">{header}</div>
        </div>
        {headerRight && (
          <div className="flex-shrink-0 ms-2" onClick={(e) => e.stopPropagation()}>
            {headerRight}
          </div>
        )}
      </div>
      {isOpen && <div className="accordion-body p-2">{children}</div>}
    </div>
  );
}

// Simple standalone accordion for use without a group
interface SimpleAccordionProps {
  header: ReactNode;
  headerRight?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}

export function SimpleAccordion({
  header,
  headerRight,
  children,
  defaultOpen = false,
  className = "",
}: SimpleAccordionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={`border rounded ${className}`}>
      <div
        className="d-flex align-items-center justify-content-between px-2 py-1 bg-body-tertiary"
        style={{ cursor: "pointer", minHeight: "32px" }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="d-flex align-items-center gap-1 flex-grow-1 overflow-hidden">
          {isOpen ? (
            <ChevronDown size={14} className="flex-shrink-0 text-muted" />
          ) : (
            <ChevronRight size={14} className="flex-shrink-0 text-muted" />
          )}
          <div className="overflow-hidden">{header}</div>
        </div>
        {headerRight && (
          <div className="flex-shrink-0 ms-2" onClick={(e) => e.stopPropagation()}>
            {headerRight}
          </div>
        )}
      </div>
      {isOpen && <div className="p-2">{children}</div>}
    </div>
  );
}

export default SimpleAccordion;
