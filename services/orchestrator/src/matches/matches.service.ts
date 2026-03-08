import { Injectable, Logger } from '@nestjs/common';
import { PrismaClient, EventType } from '@prisma/client';
import type { Match } from '@matcha/database';
import { HttpService } from '@nestjs/axios';
import { EventsGateway } from '../events/events.gateway';
import { firstValueFrom } from 'rxjs';
import * as fs from 'fs';
import * as path from 'path';
import 'multer';
import { WsEvents, isYoutubeUrl, AnalysisPayload } from '@matcha/shared';

@Injectable()
export class MatchesService {
  private prisma: PrismaClient;
  private readonly logger = new Logger(MatchesService.name);

  constructor(
    private eventsGateway: EventsGateway,
    private httpService: HttpService,
  ) {
    this.prisma = new PrismaClient();
  }

  async create(file: Express.Multer.File, userId: string): Promise<Match> {
    if (!file || !file.originalname) {
      throw new Error('Invalid file upload');
    }
    if (file.size > 5000000000) {
      throw new Error('File exceeds 5GB limit');
    }

    let filePath: string;
    let fileName: string;

    if (file.path) {
      // Multer diskStorage saved the file
      filePath = file.path;
      fileName = file.filename;
    } else if (file.buffer) {
      // Fallback for memoryStorage
      fileName = `${Date.now()}-${file.originalname}`;
      const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');

      try {
        if (!fs.existsSync(uploadsDir)) {
          fs.mkdirSync(uploadsDir, { recursive: true });
        }
        filePath = path.join(uploadsDir, fileName);
        fs.writeFileSync(filePath, file.buffer);
      } catch (error) {
        this.logger.error(`Failed to save upload: ${(error as Error).message}`);
        throw error;
      }
    } else {
      throw new Error('Invalid file upload: neither buffer nor path found');
    }

    // Use the actual filesystem path (works for native Windows execution)
    const publicUrl = `/uploads/${fileName}`;

    const match = await this.prisma.match.create({
      data: {
        uploadUrl: publicUrl,
        status: 'UPLOADED',
        duration: 0,
        userId,
      },
    });

    void this.triggerInference(match.id, filePath);

    return match;
  }

  async createFromYoutube(
    url: string,
    userId: string,
    startTime?: number,
    endTime?: number,
  ): Promise<Match> {
    if (!url || !isYoutubeUrl(url)) {
      throw new Error('Invalid YouTube URL');
    }

    const match = await this.prisma.match.create({
      data: {
        uploadUrl: url,
        status: 'UPLOADED',
        duration: 0,
        userId,
      },
    });

    this.logger.log(
      `Created match ${match.id} from YouTube URL: ${url} (range: ${startTime}-${endTime})`,
    );

    // Pass the raw YouTube URL and range to inference.
    void this.triggerInference(match.id, url, startTime, endTime);

    return match;
  }

  async triggerInference(
    matchId: string,
    videoUrl: string,
    startTime?: number,
    endTime?: number,
  ) {
    const inferenceUrl = process.env.INFERENCE_URL || 'http://localhost:8000';
    const maxAttempts = 5;
    const baseDelayMs = 2000;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        this.logger.log(
          `Triggering inference for match ${matchId} (attempt ${attempt}/${maxAttempts})`,
        );
        await firstValueFrom(
          this.httpService.post(
            `${inferenceUrl}/api/v1/analyze`,
            {
              match_id: matchId,
              video_url: videoUrl,
              start_time: startTime,
              end_time: endTime,

            },
            { timeout: 30000 },
          ) as any,
        );
        return; // Success
      } catch (error) {
        const isLast = attempt === maxAttempts;
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        this.logger.error(
          `Inference trigger attempt ${attempt}/${maxAttempts} failed: ${errorMessage}`,
        );
        if (isLast) {
          // Mark match as FAILED so it's visible in UI
          await this.prisma.match
            .update({ where: { id: matchId }, data: { status: 'FAILED' } })
            .catch(() => {});
          this.eventsGateway.server
            .to(matchId)
            .emit(WsEvents.PROGRESS, { matchId, progress: -1 });
          return;
        }
        // Exponential backoff: 2s, 4s, 8s, 16s
        const delay = baseDelayMs * Math.pow(2, attempt - 1);
        this.logger.log(`Retrying in ${delay}ms...`);
      }
    }
  }

  async findAll(userId?: string) {
    const where = userId ? { OR: [{ userId }, { userId: null }] } : {};

    const matches = await this.prisma.match.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      include: {
        _count: { select: { events: true, highlights: true } },
      },
    });
    return matches;
  }

  async findOne(id: string, userId?: string) {
    const where = userId
      ? { id, OR: [{ userId }, { userId: null }] }
      : { id, userId: null };

    return this.prisma.match.findFirst({
      where,
      include: {
        events: { orderBy: { timestamp: 'asc' } },
        highlights: { orderBy: { startTime: 'asc' } },
        emotionScores: { orderBy: { timestamp: 'asc' } },
      },
    });
  }

  addLiveEvent(id: string, event: any) {
    /**
     * Called by the inference service for EACH detected event immediately,
     * before the full analysis completes.  We broadcast it via WebSocket so
     * the browser can populate the event feed in real-time.
     * We do NOT save to DB here – the final complete() call saves everything.
     */
    this.eventsGateway.server.to(id).emit(WsEvents.MATCH_EVENT, {
      matchId: id,
      event,
    });
    return { ok: true };
  }

  async pushTrackingUpdate(id: string, frames: any[]) {
    /**
     * Called by the inference service periodically with newly-tracked frames.
     * Broadcasts via WebSocket so the browser overlay updates in real-time
     * without waiting for the full analysis to complete.
     */
    if (!frames?.length) return { ok: true };
    this.eventsGateway.server.to(id).emit(WsEvents.TRACKING_UPDATE, {
      matchId: id,
      frames,
    });
    return { ok: true };
  }

  async updateProgress(id: string, progress: number, stage?: string) {
    // progress === -1 is a failure signal from the inference service
    const status = progress === -1 ? 'FAILED' : 'PROCESSING';
    const safeProgress = progress === -1 ? 0 : Math.round(progress);

    await this.prisma.match
      .update({
        where: { id },
        data: { status, progress: safeProgress },
      })
      .catch((err: Error) => {
        this.logger.warn(
          `Failed to update progress for match ${id}: ${err.message}`,
        );
      });
    this.eventsGateway.server
      .to(id)
      .emit(WsEvents.PROGRESS, { matchId: id, progress, stage: stage || undefined });
  }

  async completeMatch(id: string, payload: AnalysisPayload) {
    if (!id || !payload) {
      throw new Error('Invalid match ID or payload');
    }

    const {
      events = [],
      highlights = [],
      emotionScores = [],
      duration,
      summary,
      highlightReelUrl,
      highlightReelPortraitUrl,
      trackingData,
      teamColors,
      heatmapUrl,
      topSpeedKmh,
      videoUrl,
      thumbnailUrl,
    } = payload;

    const typeMap: Record<string, EventType> = {
      GOAL: EventType.GOAL,
      FOUL: EventType.FOUL,
      TACKLE: EventType.TACKLE,
      SAVE: EventType.SAVE,
      CELEBRATION: EventType.CELEBRATION,
      Celebrate: EventType.CELEBRATION, // Legacy fallback for old code
      HIGHLIGHT: EventType.HIGHLIGHT,
      PENALTY: EventType.PENALTY,
      RED_CARD: EventType.RED_CARD,
      YELLOW_CARD: EventType.YELLOW_CARD,
      CORNER: EventType.CORNER,
      OFFSIDE: EventType.OFFSIDE,
    };
    const validTypes = new Set<string>(Object.values(EventType));

    const validEvents = events
      .map((e) => ({
        matchId: id,
        timestamp: e.timestamp,
        type: typeMap[e.type] ?? EventType.HIGHLIGHT, // Safe fallback to generic highlight
        confidence: Math.max(0, Math.min(1, e.confidence)), // Clamp 0-1
        finalScore: Math.max(0, Math.min(10, e.finalScore ?? 0)), // Clamp 0-10
        commentary: (e.commentary ?? '').substring(0, 1000), // Cap at 1000 chars
      }))
      .filter((e) => validTypes.has(e.type));

    const validHighlights = (highlights ?? []).map((h) => ({
      matchId: id,
      startTime: h.startTime,
      endTime: h.endTime,
      score: Math.max(0, Math.min(10, h.score)), // Clamp 0-10
      eventType: (h.eventType ?? '').substring(0, 50), // Cap at 50 chars
      commentary: (h.commentary ?? '').substring(0, 500), // Cap at 500 chars
      videoUrl: h.videoUrl ?? null,
    }));

    const validEmotion = (emotionScores ?? []).map((s) => ({
      matchId: id,
      timestamp: s.timestamp,
      audioScore: Math.max(0, Math.min(1, s.audioScore)), // Clamp 0-1
      motionScore: Math.max(0, Math.min(1, s.motionScore)), // Clamp 0-1
      contextWeight: Math.max(0, Math.min(1, s.contextWeight)), // Clamp 0-1
      finalScore: Math.max(0, Math.min(10, s.finalScore)), // Clamp 0-10
    }));

    await this.prisma.$transaction([
      this.prisma.event.createMany({ data: validEvents }),
      this.prisma.highlight.createMany({ data: validHighlights }),
      this.prisma.emotionScore.createMany({ data: validEmotion }),
      this.prisma.match.update({
        where: { id },
        data: {
          status: 'COMPLETED',
          duration: Math.max(0, duration ?? 0), // Ensure non-negative
          summary: (summary ?? '').substring(0, 5000), // Cap at 5000 chars
          highlightReelUrl,
          highlightReelPortraitUrl: highlightReelPortraitUrl ?? null,
          trackingData,
          teamColors,
          heatmapUrl: heatmapUrl ?? null,
          thumbnailUrl: thumbnailUrl ?? null,
          topSpeedKmh: topSpeedKmh ?? null,
          ...(videoUrl ? { uploadUrl: videoUrl } : {}),
        } as any,
      }),
    ]);

    this.eventsGateway.server
      .to(id)
      .emit(WsEvents.PROGRESS, { matchId: id, progress: 100, stage: 'done' });
    this.eventsGateway.server.to(id).emit(WsEvents.COMPLETE, {
      matchId: id,
      eventCount: validEvents.length,
      highlightCount: validHighlights.length,
    });

    this.logger.log(
      `Match ${id} completed — ${validEvents.length} events, ${validHighlights.length} highlights.`,
    );
    return { ok: true };
  }

  async reanalyzeMatch(id: string, userId: string): Promise<{ ok: boolean }> {
    if (!id || typeof id !== 'string') {
      this.logger.error('Invalid match ID for reanalysis');
      return { ok: false };
    }

    const match = await this.prisma.match.findFirst({
      where: { id, OR: [{ userId }, { userId: null }] },
    });
    if (!match) {
      this.logger.warn(`Match not found: ${id}`);
      return { ok: false };
    }

    // Wipe previous analysis results, keep the video source
    await this.prisma.$transaction([
      this.prisma.event.deleteMany({ where: { matchId: id } }),
      this.prisma.highlight.deleteMany({ where: { matchId: id } }),
      this.prisma.emotionScore.deleteMany({ where: { matchId: id } }),
      this.prisma.match.update({
        where: { id },
        data: {
          status: 'PROCESSING',
          progress: 0,
          trackingData: undefined,
          teamColors: undefined,
          heatmapUrl: null,
          thumbnailUrl: null,
          topSpeedKmh: null,
          summary: null,
          duration: null,
        } as any,
      }),
    ]);

    const uploadUrl = match.uploadUrl;

    // If the stored URL is a YouTube/external URL, pass it directly — no file path reconstruction
    if (
      uploadUrl.startsWith('http://youtube.com') ||
      uploadUrl.startsWith('https://youtube.com') ||
      uploadUrl.startsWith('https://youtu.be') ||
      uploadUrl.startsWith('http://youtu.be') ||
      uploadUrl.includes('youtube.com')
    ) {
      this.logger.log(`Re-analysing YouTube match ${id}: ${uploadUrl}`);
      void this.triggerInference(id, uploadUrl);
      return { ok: true };
    }

    // For uploaded files: reconstruct the filesystem path from the stored public URL
    // uploadUrl format: http://localhost:4000/uploads/<filename>
    const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');
    const uploadUrlParts = uploadUrl.split('/uploads/');
    if (uploadUrlParts.length < 2) {
      this.logger.error(
        `Cannot reconstruct file path from uploadUrl: ${uploadUrl}`,
      );
      await this.prisma.match
        .update({ where: { id }, data: { status: 'FAILED' } as any })
        .catch(() => {});
      return { ok: false };
    }
    const fileName = uploadUrlParts[uploadUrlParts.length - 1];
    const videoPath = path.join(uploadsDir, fileName);

    void this.triggerInference(id, videoPath);
    return { ok: true };
  }

  async deleteMatch(id: string, userId: string): Promise<{ ok: boolean }> {
    const match = await this.prisma.match.findFirst({
      where: { id, OR: [{ userId }, { userId: null }] },
    });
    if (!match) return { ok: false };

    await this.prisma.$transaction([
      this.prisma.event.deleteMany({ where: { matchId: id } }),
      this.prisma.highlight.deleteMany({ where: { matchId: id } }),
      this.prisma.emotionScore.deleteMany({ where: { matchId: id } }),
      this.prisma.match.delete({ where: { id } }),
    ]);
    this.logger.log(`Match ${id} deleted.`);
    return { ok: true };
  }

  // ── Highlight-level operations ───────────────────────────────────────

  async updateHighlight(
    matchId: string,
    highlightId: string,
    data: {
      startTime?: number;
      endTime?: number;
      eventType?: string;
      score?: number;
      commentary?: string;
    },
    userId: string,
  ) {
    // Verify match ownership
    const match = await this.prisma.match.findFirst({
      where: { id: matchId, OR: [{ userId }, { userId: null }] },
    });
    if (!match) return { ok: false };

    const updateData: Record<string, unknown> = {};
    if (data.startTime !== undefined) updateData.startTime = data.startTime;
    if (data.endTime !== undefined) updateData.endTime = data.endTime;
    if (data.eventType !== undefined)
      updateData.eventType = data.eventType.substring(0, 50);
    if (data.score !== undefined)
      updateData.score = Math.max(0, Math.min(10, data.score));
    if (data.commentary !== undefined)
      updateData.commentary = (data.commentary ?? '').substring(0, 500);

    const updated = await this.prisma.highlight.update({
      where: { id: highlightId },
      data: updateData,
    });
    this.logger.log(
      `Highlight ${highlightId} updated on match ${matchId}`,
    );
    return updated;
  }

  async deleteHighlight(
    matchId: string,
    highlightId: string,
    userId: string,
  ): Promise<{ ok: boolean }> {
    const match = await this.prisma.match.findFirst({
      where: { id: matchId, OR: [{ userId }, { userId: null }] },
    });
    if (!match) return { ok: false };

    await this.prisma.highlight.delete({ where: { id: highlightId } });
    this.logger.log(
      `Highlight ${highlightId} rejected/deleted from match ${matchId}`,
    );
    return { ok: true };
  }

  async getGlobalStats() {
    const [matchesCount, eventsCount, highlightsCount, durationAgg] = await Promise.all([
      this.prisma.match.count({ where: { status: 'COMPLETED' } }),
      this.prisma.event.count(),
      this.prisma.highlight.count(),
      this.prisma.match.aggregate({
        _sum: { duration: true },
        where: { status: 'COMPLETED' },
      }),
    ]);

    return {
      totalMatches: matchesCount,
      totalEvents: eventsCount,
      totalHighlights: highlightsCount,
      totalDuration: durationAgg._sum.duration || 0,
    };
  }
}
