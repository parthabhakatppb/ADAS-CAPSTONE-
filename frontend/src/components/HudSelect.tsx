import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Check } from 'lucide-react';

export interface HudSelectOption {
  value: string;
  label: string;
  icon?: React.ReactNode;
  sublabel?: string;
}

interface HudSelectProps {
  value: string;
  options: HudSelectOption[];
  onChange: (value: string) => void;
  icon?: React.ReactNode;
  placement?: 'bottom' | 'top';
  align?: 'left' | 'right';
  className?: string;
  menuClassName?: string;
  ariaLabel?: string;
  mono?: boolean;
}

const HudSelect: React.FC<HudSelectProps> = ({
  value,
  options,
  onChange,
  icon,
  placement = 'bottom',
  align = 'left',
  className = '',
  menuClassName = '',
  ariaLabel,
  mono = false,
}) => {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number>(() =>
    Math.max(0, options.findIndex(o => o.value === value))
  );
  const [coords, setCoords] = useState<{ top: number; left: number; minWidth: number }>({
    top: 0,
    left: 0,
    minWidth: 140,
  });

  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const selected = options.find(o => o.value === value) ?? options[0];

  const updatePosition = useCallback(() => {
    const btn = triggerRef.current;
    if (!btn) return;
    const rect = btn.getBoundingClientRect();
    const minWidth = Math.max(rect.width, 140);
    const left =
      align === 'right'
        ? Math.max(8, rect.right - minWidth)
        : Math.min(rect.left, window.innerWidth - minWidth - 8);

    const top = placement === 'top' ? rect.top - 6 : rect.bottom + 6;
    setCoords({ top, left, minWidth });
  }, [align, placement]);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    setActiveIndex(Math.max(0, options.findIndex(o => o.value === value)));

    const handlePointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target) || menuRef.current?.contains(target)) {
        return;
      }
      setOpen(false);
    };

    const handleResize = () => setOpen(false);

    document.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('resize', handleResize);
    window.addEventListener('scroll', handleResize, true);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('scroll', handleResize, true);
    };
  }, [open, options, updatePosition, value]);

  const selectOption = (val: string) => {
    onChange(val);
    setOpen(false);
    triggerRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }

    if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex(prev => (prev + 1) % options.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex(prev => (prev - 1 + options.length) % options.length);
    } else if (e.key === 'Home') {
      e.preventDefault();
      setActiveIndex(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      setActiveIndex(options.length - 1);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const opt = options[activeIndex];
      if (opt) selectOption(opt.value);
    }
  };

  const displayIcon = selected?.icon ?? icon;

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen(prev => !prev)}
        onKeyDown={handleKeyDown}
        className={`group inline-flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-md text-xs font-medium border transition-all select-none focus:outline-none ${
          open
            ? 'bg-drivable/15 text-white border-drivable/50 shadow-[0_0_0_1px_rgba(6,182,212,0.2)]'
            : 'bg-white/5 text-gray-200 border-border hover:bg-white/10 hover:text-white hover:border-white/20'
        } ${mono ? 'font-mono tabular-nums' : ''} ${className}`}
      >
        <span className="flex items-center gap-1.5 min-w-0 truncate">
          {displayIcon && (
            <span className={`flex-shrink-0 ${open ? 'text-drivable' : 'text-gray-400 group-hover:text-drivable'} transition-colors`}>
              {displayIcon}
            </span>
          )}
          <span className="truncate">{selected?.label ?? value}</span>
        </span>
        <ChevronDown
          className={`w-3.5 h-3.5 flex-shrink-0 transition-transform duration-150 ${
            open ? 'rotate-180 text-drivable' : 'text-gray-400 group-hover:text-gray-200'
          }`}
        />
      </button>

      {open &&
        createPortal(
          <div
            ref={menuRef}
            role="listbox"
            aria-label={ariaLabel}
            style={{
              position: 'fixed',
              top: coords.top,
              left: coords.left,
              minWidth: coords.minWidth,
              transform: placement === 'top' ? 'translateY(-100%)' : undefined,
            }}
            className={`z-50 bg-card/95 backdrop-blur-md border border-border rounded-lg p-1 shadow-2xl shadow-black/80 ring-1 ring-white/5 ${menuClassName}`}
          >
            {options.map((opt, idx) => {
              const isSelected = opt.value === value;
              const isHighlighted = idx === activeIndex;
              return (
                <button
                  key={opt.value}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onMouseEnter={() => setActiveIndex(idx)}
                  onClick={() => selectOption(opt.value)}
                  className={`w-full flex items-center justify-between gap-2.5 px-2.5 py-1.5 rounded-md text-left text-xs transition-colors ${
                    isSelected
                      ? 'bg-drivable/15 text-drivable font-semibold'
                      : isHighlighted
                        ? 'bg-white/10 text-white'
                        : 'text-gray-300 hover:bg-white/5 hover:text-white'
                  } ${mono ? 'font-mono tabular-nums' : ''}`}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    {opt.icon && (
                      <span className={`flex-shrink-0 ${isSelected ? 'text-drivable' : 'text-gray-400'}`}>
                        {opt.icon}
                      </span>
                    )}
                    <span className="flex flex-col min-w-0">
                      <span className="truncate">{opt.label}</span>
                      {opt.sublabel && (
                        <span className="text-[10px] text-gray-500 font-normal truncate">
                          {opt.sublabel}
                        </span>
                      )}
                    </span>
                  </span>
                  <Check
                    className={`w-3.5 h-3.5 flex-shrink-0 transition-opacity ${
                      isSelected ? 'opacity-100 text-drivable' : 'opacity-0'
                    }`}
                  />
                </button>
              );
            })}
          </div>,
          document.body
        )}
    </>
  );
};

export default HudSelect;
