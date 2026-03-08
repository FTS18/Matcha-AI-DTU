import {
  Controller,
  Get,
  Post,
  Patch,
  Delete,
  Param,
  Body,
  UseInterceptors,
  UploadedFile,
  NotFoundException,
  BadRequestException,
  UseGuards,
  Req,
  Query,
} from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { OptionalJwtAuthGuard } from '../auth/optional-jwt-auth.guard';
import { FileInterceptor } from '@nestjs/platform-express';
import { Throttle } from '@nestjs/throttler';
import { diskStorage } from 'multer';
import { extname, join } from 'path';
import { MatchesService } from './matches.service';
import type { Match } from '@matcha/database';
import type { AnalysisPayload } from '@matcha/shared';
import 'multer';

interface AuthRequestOptional extends Express.Request {
  user?: {
    userId: string;
  };
}

interface AuthRequestRequired extends Express.Request {
  user: {
    userId: string;
  };
}

@Controller('matches')
export class MatchesController {
  constructor(private readonly matchesService: MatchesService) {}

  // Stricter rate limit on upload — 5 uploads per minute to protect disk + inference queue
  @UseGuards(JwtAuthGuard)
  @Throttle({ default: { ttl: 60_000, limit: 5 } })
  @Post('upload')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: join(process.cwd(), '..', '..', 'uploads'),
        filename: (req, file, cb) => {
          const uniqueSuffix =
            Date.now() + '-' + Math.round(Math.random() * 1e9);
          cb(null, `${uniqueSuffix}${extname(file.originalname)}`);
        },
      }),
      limits: {
        fileSize: 5 * 1024 * 1024 * 1024, // 5GB limit
      },
    }),
  )
  async uploadFile(
    @UploadedFile() file: Express.Multer.File,
    @Req() req: AuthRequestRequired,
  ): Promise<Match> {
    if (!file) throw new BadRequestException('No file provided');
    return this.matchesService.create(file, req.user.userId);
  }

  @UseGuards(JwtAuthGuard)
  @Get('yt-info')
  async ytInfo(@Query('url') url: string) {
    if (!url) throw new BadRequestException('url query param required');
    try {
      const parsed = new URL(url);
      if (!parsed.hostname.includes('youtube.com') && !parsed.hostname.includes('youtu.be')) {
        throw new BadRequestException('Must be a YouTube URL');
      }
    } catch (e) {
      if (e instanceof BadRequestException) throw e;
      throw new BadRequestException('Invalid URL');
    }
    // Proxy to inference service
    const inferenceUrl = process.env.INFERENCE_URL || 'http://localhost:8000';
    try {
      const resp = await fetch(`${inferenceUrl}/api/v1/yt-info?url=${encodeURIComponent(url)}`);
      if (!resp.ok) throw new Error(await resp.text());
      return resp.json();
    } catch (e: any) {
      throw new BadRequestException(`Could not fetch video info: ${e.message}`);
    }
  }

  @UseGuards(JwtAuthGuard)
  @Throttle({ default: { ttl: 60_000, limit: 10 } })
  @Post('youtube')
  async uploadYoutube(
    @Body() body: { url: string; start_time?: number; end_time?: number },
    @Req() req: AuthRequestRequired,
  ): Promise<Match> {
    if (!body || !body.url) {
      throw new BadRequestException('YouTube URL is required');
    }
    // Validate URL format
    try {
      const url = new URL(body.url);
      if (
        !url.hostname.includes('youtube.com') &&
        !url.hostname.includes('youtu.be')
      ) {
        throw new BadRequestException(
          'URL must be a YouTube link (youtube.com or youtu.be)',
        );
      }
    } catch (e) {
      if (e instanceof BadRequestException) throw e;
      throw new BadRequestException('Invalid URL format');
    }
    // Validate time range if provided
    if (body.start_time !== undefined && body.start_time < 0) {
      throw new BadRequestException('start_time must be >= 0');
    }
    if (body.end_time !== undefined && body.end_time <= 0) {
      throw new BadRequestException('end_time must be > 0');
    }
    if (
      body.start_time !== undefined &&
      body.end_time !== undefined &&
      body.start_time >= body.end_time
    ) {
      throw new BadRequestException('start_time must be less than end_time');
    }
    return this.matchesService.createFromYoutube(
      body.url,
      req.user.userId,
      body.start_time,
      body.end_time,
    );
  }

  @Get('stats')
  async getStats() {
    return this.matchesService.getGlobalStats();
  }

  @UseGuards(OptionalJwtAuthGuard)
  @Get()
  async findAll(@Req() req: AuthRequestOptional): Promise<Match[]> {
    return this.matchesService.findAll(req.user?.userId);
  }

  @UseGuards(OptionalJwtAuthGuard)
  @Get(':id')
  async findOne(
    @Param('id') id: string,
    @Req() req: AuthRequestOptional,
  ): Promise<Match> {
    const match = await this.matchesService.findOne(id, req.user?.userId);
    if (!match) throw new NotFoundException(`Match ${id} not found`);
    return match;
  }

  @Post(':id/progress')
  async updateProgress(
    @Param('id') id: string,
    @Body() body: { progress: number; stage?: string },
  ) {
    if (typeof body.progress !== 'number') {
      throw new BadRequestException('progress must be a number');
    }
    return this.matchesService.updateProgress(id, body.progress, body.stage);
  }

  @Post(':id/live-event')
  addLiveEvent(@Param('id') id: string, @Body() body: object) {
    return this.matchesService.addLiveEvent(id, body);
  }

  @Post(':id/tracking-update')
  async trackingUpdate(@Param('id') id: string, @Body() body: { frames: object[] }) {
    return this.matchesService.pushTrackingUpdate(id, body?.frames ?? []);
  }

  @Post(':id/complete')
  async completeMatch(@Param('id') id: string, @Body() body: AnalysisPayload) {
    if (!body) throw new BadRequestException('Payload required');
    return this.matchesService.completeMatch(id, body);
  }

  @UseGuards(JwtAuthGuard)
  @Post(':id/reanalyze')
  async reanalyzeMatch(
    @Param('id') id: string,
    @Req() req: AuthRequestRequired,
  ): Promise<{ ok: boolean }> {
    const result = await this.matchesService.reanalyzeMatch(
      id,
      req.user.userId,
    );
    if (!result.ok) throw new NotFoundException(`Match ${id} not found`);
    return result;
  }

  @UseGuards(JwtAuthGuard)
  @Delete(':id')
  async deleteMatch(
    @Param('id') id: string,
    @Req() req: AuthRequestRequired,
  ): Promise<{ ok: boolean }> {
    return this.matchesService.deleteMatch(id, req.user.userId);
  }

  // ── Highlight-level endpoints ──────────────────────────────────────────

  @UseGuards(JwtAuthGuard)
  @Patch(':matchId/highlights/:highlightId')
  async updateHighlight(
    @Param('matchId') matchId: string,
    @Param('highlightId') highlightId: string,
    @Body()
    body: {
      startTime?: number;
      endTime?: number;
      eventType?: string;
      score?: number;
      commentary?: string;
    },
    @Req() req: AuthRequestRequired,
  ) {
    return this.matchesService.updateHighlight(
      matchId,
      highlightId,
      body,
      req.user.userId,
    );
  }

  @UseGuards(JwtAuthGuard)
  @Delete(':matchId/highlights/:highlightId')
  async deleteHighlight(
    @Param('matchId') matchId: string,
    @Param('highlightId') highlightId: string,
    @Req() req: AuthRequestRequired,
  ): Promise<{ ok: boolean }> {
    return this.matchesService.deleteHighlight(
      matchId,
      highlightId,
      req.user.userId,
    );
  }
}
