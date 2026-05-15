// @ts-nocheck
import React from "react";
import { Document, Page, Text, View, Image, StyleSheet } from "@react-pdf/renderer";
import { MatchEvent, Highlight, formatTime, EVENT_CONFIG, DEFAULT_EVENT_CONFIG } from "@matcha/shared";
import { BRAND_COLORS } from "@matcha/theme";

export interface MatchReportData {
  id: string;
  status: string;
  duration: number;
  summary?: string;
  createdAt: string;
  events: MatchEvent[];
  highlights: Highlight[];
  teamColors?: number[][];
  heatmapUrl?: string;
  topSpeedKmh?: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function rgbToHex(r: number, g: number, b: number): string {
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

const THEME_TO_HEX: Record<string, string> = {
  success: BRAND_COLORS.success,
  warning: BRAND_COLORS.warning,
  error: BRAND_COLORS.error,
  info: BRAND_COLORS.info,
  accent: BRAND_COLORS.primary,
  neutral: BRAND_COLORS.muted,
};

function getEventColor(type: string): string {
  const cfg = EVENT_CONFIG[type] || DEFAULT_EVENT_CONFIG;
  return THEME_TO_HEX[cfg.theme] || THEME_TO_HEX.neutral;
}

function getEventLabel(type: string): string {
  const cfg = EVENT_CONFIG[type] || DEFAULT_EVENT_CONFIG;
  return cfg.label;
}

function scoreColor(score: number): string {
  if (score >= 7.5) return THEME_TO_HEX.success;
  if (score >= 5) return THEME_TO_HEX.warning;
  return THEME_TO_HEX.neutral;
}

// ── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  page: {
    backgroundColor: BRAND_COLORS.background,
    color: BRAND_COLORS.text,
    fontFamily: "Helvetica",
    paddingBottom: 50,
  },

  header: {
    backgroundColor: "#0a0b12",
    borderBottomWidth: 1,
    borderBottomColor: BRAND_COLORS.border,
    paddingHorizontal: 32,
    paddingVertical: 18,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerWordmark: {
    fontSize: 16,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.primary,
    letterSpacing: 4,
  },
  headerLabel: {
    fontSize: 9,
    color: BRAND_COLORS.muted,
    letterSpacing: 3,
    textTransform: "uppercase",
  },

  footer: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    borderTopWidth: 1,
    borderTopColor: BRAND_COLORS.border,
    paddingHorizontal: 32,
    paddingVertical: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  footerText: { fontSize: 7, color: BRAND_COLORS.muted, letterSpacing: 1 },

  section: { marginHorizontal: 32, marginTop: 24 },
  sectionTitle: {
    fontSize: 8,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.primary,
    letterSpacing: 3,
    textTransform: "uppercase",
    borderBottomWidth: 1,
    borderBottomColor: BRAND_COLORS.border,
    paddingBottom: 6,
    marginBottom: 14,
  },

  metaRow: { flexDirection: "row", gap: 8, marginTop: 4 },
  metaChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: BRAND_COLORS.card,
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
  },
  metaLabel: {
    fontSize: 7,
    color: BRAND_COLORS.muted,
    letterSpacing: 2,
    textTransform: "uppercase",
  },
  metaValue: {
    fontSize: 9,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.text,
  },
  statGrid: { flexDirection: "row", gap: 10, marginTop: 2 },
  statBox: {
    flex: 1,
    backgroundColor: BRAND_COLORS.card,
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
    padding: 14,
    alignItems: "flex-start",
  },
  statLabel: {
    fontSize: 7,
    color: BRAND_COLORS.muted,
    letterSpacing: 2,
    textTransform: "uppercase",
    marginBottom: 4,
  },
  statValue: {
    fontSize: 26,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.text,
    lineHeight: 1,
  },
  statSub: { fontSize: 7, color: BRAND_COLORS.muted, marginTop: 3 },

  summaryBox: {
    backgroundColor: BRAND_COLORS.card,
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
    borderLeftWidth: 3,
    borderLeftColor: BRAND_COLORS.primary,
    padding: 16,
    marginTop: 2,
  },
  summaryText: { fontSize: 9.5, color: BRAND_COLORS.text, lineHeight: 1.7 },

  analyticsRow: { flexDirection: "row", gap: 12 },
  speedBox: {
    flex: 1,
    backgroundColor: BRAND_COLORS.card,
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
    padding: 16,
  },
  speedValue: {
    fontSize: 42,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.warning,
    lineHeight: 1,
  },
  speedUnit: {
    fontSize: 9,
    color: BRAND_COLORS.muted,
    letterSpacing: 3,
    marginTop: 4,
  },
  speedSub: {
    fontSize: 7.5,
    color: BRAND_COLORS.muted,
    marginTop: 8,
    lineHeight: 1.6,
  },
  colorBox: {
    flex: 1,
    backgroundColor: BRAND_COLORS.card,
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
    padding: 16,
  },
  swatch: { height: 28, width: 28, marginRight: 10 },
  swatchRow: { flexDirection: "row", alignItems: "center", marginBottom: 10 },
  swatchLabel: {
    fontSize: 8,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.muted,
    letterSpacing: 2,
    textTransform: "uppercase",
  },
  swatchHex: {
    fontSize: 9,
    color: BRAND_COLORS.text,
    fontFamily: "Helvetica-Bold",
  },
  heatmapBox: {
    backgroundColor: BRAND_COLORS.card,
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
    padding: 12,
    marginTop: 12,
  },
  heatmapImg: { width: "100%", height: 200, objectFit: "contain" },
  heatmapSub: {
    fontSize: 7,
    color: BRAND_COLORS.muted,
    marginTop: 8,
    textAlign: "center",
    letterSpacing: 1,
  },

  tableHeader: {
    flexDirection: "row",
    backgroundColor: "#0d0e16",
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  tableRow: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderBottomColor: BRAND_COLORS.border,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  tableRowAlt: { backgroundColor: BRAND_COLORS.card },
  colMin: { width: 45, fontSize: 8 },
  colType: { width: 90, fontSize: 8 },
  colScore: { width: 50, fontSize: 8 },
  colComm: { flex: 1, fontSize: 7.5 },
  thText: {
    fontSize: 7,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.muted,
    letterSpacing: 2,
    textTransform: "uppercase",
  },

  hlCard: {
    backgroundColor: BRAND_COLORS.card,
    borderWidth: 1,
    borderColor: BRAND_COLORS.border,
    borderLeftWidth: 3,
    padding: 12,
    marginBottom: 8,
  },
  hlRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 5,
  },
  hlTime: {
    fontSize: 8,
    fontFamily: "Helvetica-Bold",
    color: BRAND_COLORS.text,
  },
  hlType: { fontSize: 7, letterSpacing: 2, textTransform: "uppercase" },
  hlScore: { fontSize: 9, fontFamily: "Helvetica-Bold" },
  hlComm: {
    fontSize: 8,
    color: BRAND_COLORS.muted,
    lineHeight: 1.6,
    marginTop: 4,
  },
  scoreBar: { height: 3, backgroundColor: BRAND_COLORS.border, marginTop: 6 },
  scoreBarFill: { height: 3 },
});

// ── Reusable page wrapper ─────────────────────────────────────────────────────
function PageWrapper({ children, pageNum, total }: { children: React.ReactNode; pageNum: number; total: number }) {
  const now = new Date().toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
  return (
    <Page size="A4" style={styles.page}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerWordmark}>MATCHA AI</Text>
        <Text style={styles.headerLabel}>Match Report</Text>
      </View>

      {/* Content */}
      <View style={{ flex: 1 }}>{children}</View>

      {/* Footer */}
      <View style={styles.footer} fixed>
        <Text style={styles.footerText}>Generated by Matcha AI · {now}</Text>
        <Text style={styles.footerText}>
          Page {pageNum} of {total}
        </Text>
      </View>
    </Page>
  );
}

// ── The PDF Document ──────────────────────────────────────────────────────────
export function MatchReportPDF({ data }: { data: MatchReportData }) {
  const date = new Date(data.createdAt).toLocaleDateString("en-GB", {
    weekday: "long",
    day: "2-digit",
    month: "long",
    year: "numeric",
  });

  const topEvents = [...data.events].sort((a, b) => b.finalScore - a.finalScore).slice(0, 30);

  const teamA = data.teamColors?.[0];
  const teamB = data.teamColors?.[1];
  const hexA = teamA ? rgbToHex(teamA[0], teamA[1], teamA[2]) : null;
  const hexB = teamB ? rgbToHex(teamB[0], teamB[1], teamB[2]) : null;

  const totalPages = 4;

  return (
    <Document title="Match Report — Matcha AI" author="Matcha AI" creator="Matcha AI">
      <PageWrapper pageNum={1} total={totalPages}>
        <View style={styles.section}>
          <View style={styles.metaRow}>
            <View style={styles.metaChip}>
              <Text style={styles.metaLabel}>Date</Text>
              <Text style={styles.metaValue}>{date}</Text>
            </View>
            <View style={styles.metaChip}>
              <Text style={styles.metaLabel}>Duration</Text>
              <Text style={styles.metaValue}>{formatTime(data.duration)}</Text>
            </View>
            <View style={styles.metaChip}>
              <Text style={styles.metaLabel}>Status</Text>
              <Text style={[styles.metaValue, { color: BRAND_COLORS.primary }]}>{data.status}</Text>
            </View>
            <View style={styles.metaChip}>
              <Text style={styles.metaLabel}>Events</Text>
              <Text style={styles.metaValue}>{data.events.length}</Text>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Match Overview</Text>
          <View style={styles.statGrid}>
            {[
              {
                label: "Duration",
                value: formatTime(data.duration),
                sub: "match length",
              },
              {
                label: "Events",
                value: String(data.events.length),
                sub: "detected",
              },
              {
                label: "Highlights",
                value: String(data.highlights.length),
                sub: "key moments",
              },
              {
                label: "Top Score",
                value: data.events.length > 0 ? Math.max(...data.events.map((e) => e.finalScore)).toFixed(1) : "—",
                sub: "out of 10",
              },
            ].map((s) => (
              <View key={s.label} style={styles.statBox}>
                <Text style={styles.statLabel}>{s.label}</Text>
                <Text style={styles.statValue}>{s.value}</Text>
                <Text style={styles.statSub}>{s.sub}</Text>
              </View>
            ))}
          </View>
        </View>

        {data.summary && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>AI Match Summary</Text>
            <View style={styles.summaryBox}>
              <Text style={styles.summaryText}>{data.summary}</Text>
            </View>
          </View>
        )}
      </PageWrapper>

      <PageWrapper pageNum={2} total={totalPages}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Analytics</Text>

          <View style={styles.analyticsRow}>
            <View style={styles.speedBox}>
              <Text style={[styles.sectionTitle, { marginBottom: 10 }]}>Ball Speed</Text>
              <Text style={styles.speedValue}>
                {data.topSpeedKmh && data.topSpeedKmh > 0 ? data.topSpeedKmh.toFixed(1) : "—"}
              </Text>
              <Text style={styles.speedUnit}>KM / H</Text>
              <Text style={styles.speedSub}>
                Peak ball speed estimated from consecutive{"\n"}
                YOLO detections · 95th-percentile method
              </Text>
            </View>

            <View style={styles.colorBox}>
              <Text style={[styles.sectionTitle, { marginBottom: 10 }]}>Team Colours</Text>
              {hexA ? (
                <View style={styles.swatchRow}>
                  <View style={[styles.swatch, { backgroundColor: hexA }]} />
                  <View>
                    <Text style={styles.swatchLabel}>Team A</Text>
                    <Text style={styles.swatchHex}>{hexA.toUpperCase()}</Text>
                  </View>
                </View>
              ) : (
                <Text style={styles.speedSub}>Not detected</Text>
              )}
              {hexB ? (
                <View style={styles.swatchRow}>
                  <View style={[styles.swatch, { backgroundColor: hexB }]} />
                  <View>
                    <Text style={styles.swatchLabel}>Team B</Text>
                    <Text style={styles.swatchHex}>{hexB.toUpperCase()}</Text>
                  </View>
                </View>
              ) : null}
              <Text style={[styles.speedSub, { marginTop: 6 }]}>
                Auto-detected via K-Means{"\n"}clustering on jersey crops
              </Text>
            </View>
          </View>

          {data.heatmapUrl && (
            <View style={[styles.heatmapBox, { marginTop: 14 }]}>
              <Text style={[styles.sectionTitle, { marginBottom: 8 }]}>Player Density Heatmap</Text>
              <Image src={data.heatmapUrl} style={styles.heatmapImg} />
              <Text style={styles.heatmapSub}>
                Player density accumulated over entire match · Gaussian blur overlay
              </Text>
            </View>
          )}

          {!data.heatmapUrl && (
            <View style={[styles.heatmapBox, { marginTop: 14, alignItems: "center", paddingVertical: 30 }]}>
              <Text style={{ fontSize: 9, color: BRAND_COLORS.muted }}>Heatmap not available</Text>
            </View>
          )}
        </View>
      </PageWrapper>

      <PageWrapper pageNum={3} total={totalPages}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Events Timeline · {topEvents.length} events</Text>

          {topEvents.length === 0 ? (
            <View style={{ padding: 20, alignItems: "center" }}>
              <Text style={{ fontSize: 9, color: BRAND_COLORS.muted }}>No events detected.</Text>
            </View>
          ) : (
            <>
              <View style={styles.tableHeader}>
                <Text style={[styles.colMin, styles.thText]}>Min</Text>
                <Text style={[styles.colType, styles.thText]}>Event</Text>
                <Text style={[styles.colScore, styles.thText]}>Score</Text>
                <Text style={[styles.colComm, styles.thText]}>Commentary</Text>
              </View>

              {topEvents.map((ev, i) => (
                <View key={i} style={[styles.tableRow, i % 2 === 1 ? styles.tableRowAlt : {}]}>
                  <Text
                    style={[
                      styles.colMin,
                      {
                        color: BRAND_COLORS.muted,
                        fontFamily: "Helvetica-Bold",
                      },
                    ]}
                  >
                    {formatTime(ev.timestamp)}
                  </Text>
                  <Text
                    style={[
                      styles.colType,
                      {
                        color: getEventColor(ev.type),
                        fontFamily: "Helvetica-Bold",
                        letterSpacing: 0.5,
                      },
                    ]}
                  >
                    {getEventLabel(ev.type)}
                  </Text>
                  <Text
                    style={[
                      styles.colScore,
                      {
                        color: scoreColor(ev.finalScore),
                        fontFamily: "Helvetica-Bold",
                      },
                    ]}
                  >
                    {ev.finalScore.toFixed(1)}
                  </Text>
                  <Text style={[styles.colComm, { color: BRAND_COLORS.muted }]}>{ev.commentary || "—"}</Text>
                </View>
              ))}
            </>
          )}
        </View>
      </PageWrapper>

      <PageWrapper pageNum={4} total={totalPages}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Highlight Reel · {data.highlights.length} clips</Text>

          {data.highlights.length === 0 ? (
            <View style={{ padding: 20, alignItems: "center" }}>
              <Text style={{ fontSize: 9, color: BRAND_COLORS.muted }}>No highlights generated.</Text>
            </View>
          ) : (
            data.highlights
              .sort((a, b) => b.score - a.score)
              .map((hl, i) => {
                const barWidth = (hl.score / 10) * 100;
                const color = scoreColor(hl.score);
                return (
                  <View key={i} style={[styles.hlCard, { borderLeftColor: color }]}>
                    <View style={styles.hlRow}>
                      <Text style={styles.hlTime}>
                        {formatTime(hl.startTime)} → {formatTime(hl.endTime)}
                      </Text>
                      <View
                        style={{
                          flexDirection: "row",
                          alignItems: "center",
                          gap: 10,
                        }}
                      >
                        <Text style={[styles.hlType, { color }]}>
                          {hl.eventType ? getEventLabel(hl.eventType) : "Highlight"}
                        </Text>
                        <Text style={[styles.hlScore, { color }]}>{hl.score.toFixed(1)}/10</Text>
                      </View>
                    </View>

                    <View style={styles.scoreBar}>
                      <View style={[styles.scoreBarFill, { width: `${barWidth}%`, backgroundColor: color }]} />
                    </View>

                    {hl.commentary && <Text style={styles.hlComm}>{hl.commentary}</Text>}
                  </View>
                );
              })
          )}
        </View>
      </PageWrapper>
    </Document>
  );
}
