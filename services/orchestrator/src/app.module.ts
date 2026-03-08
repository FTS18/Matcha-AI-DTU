import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ServeStaticModule } from '@nestjs/serve-static';
import { ThrottlerModule } from '@nestjs/throttler';
import * as path from 'path';
import { join } from 'path';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MatchesModule } from './matches/matches.module';
import { EventsModule } from './events/events.module';
import { AuthModule } from './auth/auth.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),

    // Rate limiting — 60 requests / 60s globally. Upload endpoint adds its own tighter guard.
    ThrottlerModule.forRoot([{ ttl: 60_000, limit: 60 }]),

    ServeStaticModule.forRoot({
      rootPath: path.join(process.cwd(), '..', '..', 'uploads'),
      serveRoot: '/uploads',
      serveStaticOptions: {
        setHeaders: (res) => {
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
