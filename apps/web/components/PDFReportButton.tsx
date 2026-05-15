// @ts-nocheck
"use client";

// This file may ONLY be loaded via next/dynamic with { ssr: false }.
// It statically imports @react-pdf/renderer which is ESM-only and will crash
// if Next.js tries to evaluate it on the server.
import { PDFDownloadLink } from "@react-pdf/renderer";
import { Loader2, FileDown } from "lucide-react";
import { MatchReportPDF, type MatchReportData } from "@matcha/ui";

export default function PDFReportButton({ data }: { data: MatchReportData }) {
  return (
    <PDFDownloadLink
      document={<MatchReportPDF data={data} />}
      fileName={`matcha-match-report-${data.id.slice(0, 8)}.pdf`}
    >
      {({ loading }) => (
        <button
          disabled={loading}
          className="flex items-center gap-1.5 text-[10px] sm:text-xs text-primary hover:text-primary/80 border border-primary/40 hover:border-primary/70 hover:bg-primary/5 px-3 sm:px-4 py-1.5 transition-all duration-200 uppercase tracking-wide cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-50 disabled:cursor-wait"
          aria-label="Download match PDF report"
        >
          {loading ? <Loader2 className="size-3.5 animate-spin" /> : <FileDown className="size-3.5" />}
          <span className="hidden sm:inline">{loading ? "Building…" : "Report"}</span>
        </button>
      )}
    </PDFDownloadLink>
  );
}
