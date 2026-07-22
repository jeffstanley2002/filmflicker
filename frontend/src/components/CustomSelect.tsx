import { Check, ChevronDown } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";

export type SelectOption = {
  value: string;
  label: string;
};

type Props = {
  value: string;
  options: SelectOption[];
  ariaLabel: string;
  onChange: (value: string) => void;
};

export function CustomSelect({ value, options, ariaLabel, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const listboxId = useId();
  const selected = options.find((option) => option.value === value) ?? options[0];

  useEffect(() => {
    function close(event: MouseEvent) {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  useEffect(() => {
    if (open) optionRefs.current[activeIndex]?.focus();
  }, [activeIndex, open]);

  function openMenu() {
    setActiveIndex(Math.max(0, options.findIndex((option) => option.value === value)));
    setOpen(true);
  }

  function handleKeyDown(event: ReactKeyboardEvent) {
    if (!open && ["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
      event.preventDefault();
      openMenu();
      return;
    }
    if (!open) return;
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
      ref.current?.querySelector<HTMLButtonElement>(".select-trigger")?.focus();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      setActiveIndex((index) => (index + direction + options.length) % options.length);
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setActiveIndex(event.key === "Home" ? 0 : options.length - 1);
    }
  }

  return (
    <div className={open ? "custom-select open" : "custom-select"} ref={ref}>
      <button type="button" className="select-trigger" aria-label={ariaLabel} aria-haspopup="listbox" aria-controls={listboxId} aria-expanded={open} onKeyDown={handleKeyDown} onClick={() => open ? setOpen(false) : openMenu()}>
        <span>{selected?.label}</span>
        <ChevronDown size={20} />
      </button>
      {open ? (
        <div id={listboxId} className="select-menu" role="listbox" aria-label={ariaLabel} onKeyDown={handleKeyDown}>
          {options.map((option, index) => (
            <button
              ref={(element) => { optionRefs.current[index] = element; }}
              type="button"
              key={option.value}
              className={option.value === value ? "select-option selected" : "select-option"}
              role="option"
              aria-selected={option.value === value}
              tabIndex={index === activeIndex ? 0 : -1}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              <span>{option.label}</span>
              {option.value === value ? <Check size={17} /> : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
