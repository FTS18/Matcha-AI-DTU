import React from "react";
import { CheckCircle2, Loader2, Upload, XCircle } from "lucide-react";
import { STATUS_CONFIG as SHARED_STATUS_CONFIG } from "@matcha/shared";

export const THEME_MAP: Record<string, { color: string }> = {
  success: {
    color: "text-emerald-400 bg-emerald-500/15 border-emerald-500/40",
  },
  info: { color: "text-blue-400 bg-blue-500/15 border-blue-500/40" },
  warning: { color: "text-amber-400 bg-amber-500/15 border-amber-500/40" },
  error: { color: "text-red-400 bg-red-500/15 border-red-500/40" },
};

export const STATUS_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  COMPLETED: {
    ...SHARED_STATUS_CONFIG.COMPLETED,
    ...THEME_MAP[SHARED_STATUS_CONFIG.COMPLETED.theme],
    icon: <CheckCircle2 className="w-3 h-3" />,
  },
  PROCESSING: {
    ...SHARED_STATUS_CONFIG.PROCESSING,
    ...THEME_MAP[SHARED_STATUS_CONFIG.PROCESSING.theme],
    icon: <Loader2 className="w-3 h-3 animate-spin" />,
  },
  UPLOADED: {
    ...SHARED_STATUS_CONFIG.UPLOADED,
    ...THEME_MAP[SHARED_STATUS_CONFIG.UPLOADED.theme],
    icon: <Upload className="w-3 h-3" />,
  },
  FAILED: {
    ...SHARED_STATUS_CONFIG.FAILED,
    ...THEME_MAP[SHARED_STATUS_CONFIG.FAILED.theme],
    icon: <XCircle className="w-3 h-3" />,
  },
};
