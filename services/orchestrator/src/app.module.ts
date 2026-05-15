import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { ServeStaticModule } from '@nestjs/serve-static';
import { ThrottlerModule } from '@nestjs/throttler';
import { CacheModule } from '@nestjs/cache-manager';
import { redisStore } from 'cache-manager-redis-yet';
import * as path from 'path';
import { Response } from 'express';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MatchesModule } from './matches/matches.module';
import { EventsModule } from './events/events.module';
import { AuthModule } from './auth/auth.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),

    CacheModule.registerAsync({
      isGlobal: true,
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        store: await redisStore({
          url: configService.get('REDIS_URL') || 'redis://localhost:6379',
          ttl: 600000, // 10 minutes default
        }),
      }),
      inject: [ConfigService],
    }),

    // Rate limiting — 60 requests / 60s globally. Upload endpoint adds its own tighter guard.
    ThrottlerModule.forRoot([{ ttl: 60_000, limit: 60 }]),

    ServeStaticModule.forRoot({
      rootPath: path.join(process.cwd(), '..', '..', 'uploads'),
      serveRoot: '/uploads',
      serveStaticOptions: {
        setHeaders: (res: Response) => {
          res.set('Cross-Origin-Resource-Policy', 'cross-origin');
          res.set('Access-Control-Allow-Origin', '*');
        },
        // Disable index.html fallback for uploads directory
        index: false,
      },
    }),

    MatchesModule,
    EventsModule,
    AuthModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
