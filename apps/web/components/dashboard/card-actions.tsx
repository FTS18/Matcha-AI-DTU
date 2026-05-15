import React from "react";
import { RefreshCw, XCircle, Loader2 } from "lucide-react";

interface CardActionsProps {
  matchId: string;
  status: string;
  isConfirming: boolean;
  isDeleting: boolean;
  reanalyzingId: string | null;
  onConfirmDelete: (id: string | null) => void;
  onDelete: (id: string) => void;
  onReanalyze: (id: string) => void;
}

export const CardActions = ({
  matchId,
  status,
  isConfirming,
  isDeleting,
  reanalyzingId,
  onConfirmDelete,
  onDelete,
  onReanalyze,
}: CardActionsProps) => {
  if (isConfirming) {
    return (
      <div className="flex items-center bg-card border border-destructive/20 overflow-hidden scale-90">
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onDelete(matchId);
          }}
          disabled={isDeleting}
          className="font-mono px-3 py-2 text-[8px] bg-destructive text-white uppercase tracking-widest font-bold hover:brightness-110 flex items-center gap-1"
        >
          {isDeleting ? (
            <>
              <Loader2 className="size-2.5 animate-spin" />
              ...
            </>
          ) : (
            "DEL"
          )}
        </button>
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onConfirmDelete(null);
          }}
          className="font-mono px-3 py-2 text-[8px] text-muted-foreground hover:bg-white/5 uppercase tracking-widest border-l border-white/10"
        >
          X
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onReanalyze(matchId);
        }}
        disabled={reanalyzingId === matchId || status === "PROCESSING"}
        className={`flex items-center justify-center size-8 bg-white/5 border border-white/5 transition-all rounded-full ${
          reanalyzingId === matchId || status === "PROCESSING"
            ? "text-accent border-accent/30 cursor-wait"
            : "hover:bg-accent/10 text-muted-foreground hover:text-accent hover:border-accent/30"
        }`}
        title={status === "PROCESSING" ? "Analysis in progress" : "Reanalyze"}
      >
        <RefreshCw className={`size-3.5 ${reanalyzingId === matchId || status === "PROCESSING" ? "animate-spin" : ""}`} />
      </button>
      <button
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onConfirmDelete(matchId);
        }}
        className="flex items-center justify-center size-8 bg-white/5 hover:bg-destructive/10 text-muted-foreground hover:text-destructive border border-white/5 hover:border-destructive/30 transition-all rounded-full"
        title="Delete"
      >
        <XCircle className="size-3.5" />
      </button>
    </div>
  );
};
