"use client";

import { useEffect } from "react";
import { AlertTriangle, Home, RefreshCw } from "lucide-react";
import Link from "next/link";

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Application error boundary caught:", error);
  }, [error]);

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center text-foreground p-6">
      <div className="max-w-md w-full bg-card border border-border rounded-xl p-8 shadow-2xl flex flex-col items-center text-center">
        <div className="size-16 rounded-full bg-destructive/10 border border-destructive/20 flex items-center justify-center mb-6">
          <AlertTriangle className="size-8 text-destructive" />
        </div>

        <h2 className="font-display text-4xl mb-3">SOMETHING WENT WRONG</h2>

        <p className="font-sans text-muted-foreground mb-8">
          The Match Intelligence pipeline encountered an unexpected error while trying to render this interface.
        </p>

        <div className="w-full flex flex-col sm:flex-row gap-3">
          <button
            onClick={() => reset()}
            className="flex-1 py-3 px-4 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground font-sans text-sm font-semibold transition-colors flex items-center justify-center gap-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-background focus-visible:ring-primary"
          >
            <RefreshCw className="size-4" />
            Try again
          </button>

          <Link
            href="/"
            className="flex-1 py-3 px-4 rounded-lg border border-border bg-muted/50 hover:bg-muted font-sans text-sm font-semibold text-foreground transition-colors flex items-center justify-center gap-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-border"
          >
            <Home className="size-4" />
            Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}
