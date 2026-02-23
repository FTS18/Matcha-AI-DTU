import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getHello() {
    return {
      name: 'Matcha AI Orchestrator',
      version: '1.0.0',
      status: 'active',
      endpoints: ['/api/v1/auth', '/api/v1/matches', '/api/v1/health'],
      documentation: 'See README.md for API usage',
    };
  }
}
