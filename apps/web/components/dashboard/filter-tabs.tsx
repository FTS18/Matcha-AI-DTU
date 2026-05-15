import React from "react";
import { LayoutGrid, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";

export const FILTER_OPTIONS = ["ALL", "COMPLETED", "PROCESSING", "UPLOADED", "FAILED"] as const;
export type FilterOption = (typeof FILTER_OPTIONS)[number];

interface FilterTabsProps {
  filter: FilterOption;
  setFilter: (f: FilterOption) => void;
  matches: any[];
  isRefreshing: boolean;
  handleRefresh: () => void;
  statusConfig: any;
}

export const FilterTabs = ({
  filter,
  setFilter,
  matches,
  isRefreshing,
  handleRefresh,
  statusConfig,
}: FilterTabsProps) => {
  if (matches.length === 0) return null;

  return (
    <div className="flex gap-1 border-b border-border overflow-x-auto pb-px hide-scrollbar">
      <div className="flex items-center px-3 sm:px-4 py-2 border-r border-border bg-muted/20 shrink-0">
        <LayoutGrid className="size-3 sm:size-3.5 text-muted-foreground mr-1.5 sm:mr-2" />
        <span className="font-mono text-[9px] sm:text-[10px] text-muted-foreground uppercase tracking-[0.15em] whitespace-nowrap">
          DATA FEED
        </span>
        <button
          onClick={handleRefresh}
          className="ml-2 p-1 hover:bg-white/10 rounded-full transition-colors group"
          title="Sync Feed"
        >
          <RefreshCw
            className={cn(
              "size-3 text-muted-foreground group-hover:text-primary transition-all",
              isRefreshing && "animate-spin text-primary",
            )}
          />
        </button>
      </div>
      {FILTER_OPTIONS.map((f) => {
        const count = f === "ALL" ? matches.length : matches.filter((m) => m.status === f).length;
        const active = filter === f;
        return (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`group relative flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-5 py-2 sm:py-2.5 transition-all duration-300 shrink-0 ${
              active ? "bg-accent/10" : "hover:bg-muted/40"
            }`}
          >
            <div
              className={`font-mono text-[9px] sm:text-[10px] uppercase tracking-widest transition-colors whitespace-nowrap ${
                active ? "text-accent font-semibold" : "text-muted-foreground group-hover:text-foreground"
              }`}
            >
              {f === "ALL" ? "MASTER FEED" : (statusConfig[f]?.label ?? f)}
            </div>
            {count > 0 && (
              <div
                className={`px-1.5 py-0.5 font-mono text-[8.5px] sm:text-[9px] rounded-sm ${
                  active
                    ? "bg-accent/20 text-accent"
                    : "bg-muted text-muted-foreground group-hover:bg-border group-hover:text-foreground"
                }`}
              >
                {count}
              </div>
            )}
            {active && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent shadow-[0_-2px_8px_rgba(var(--color-accent),0.5)]" />
            )}
          </button>
        );
      })}
    </div>
  );
};
