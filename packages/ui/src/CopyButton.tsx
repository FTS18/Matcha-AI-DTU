"use client";

import React, { useState } from "react";
import { Copy, Check } from "lucide-react";

export interface CopyButtonProps {
  text: string;
  className?: string;
}

export function CopyButton({ text, className = "" }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const copy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <button
      onClick={copy}
      className={`p-1 rounded text-zinc-600 transition-all duration-200 hover:text-zinc-300 hover:bg-zinc-700 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 ${className}`}
      title="Copy to clipboard"
      aria-label="Copy to clipboard"
    >
      {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
    </button>
  );
}
