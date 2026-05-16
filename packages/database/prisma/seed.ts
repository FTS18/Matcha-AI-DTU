import { PrismaClient, MatchStatus, EventType } from '@prisma/client';
import * as bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting database seeding...');

  // 1. Create Demo User
  const hashedPassword = await bcrypt.hash('password123', 10);
  const user = await prisma.user.upsert({
    where: { email: 'demo@matcha.ai' },
    update: {},
    create: {
      email: 'demo@matcha.ai',
      password: hashedPassword,
      name: 'Demo User',
      firstName: 'Matcha',
      lastName: 'AI',
    },
  });

  console.log(`✅ Created user: ${user.email}`);

  // 2. Create Demo Matches
  const matchesData = [
    {
      uploadUrl: 'https://storage.googleapis.com/matcha-demo/match1.mp4',
      status: MatchStatus.COMPLETED,
      duration: 5400, // 90 mins
      summary: 'Thrilling 2-2 draw with a last-minute equalizer.',
      topSpeedKmh: 32.5,
      progress: 100,
      userId: user.id,
    },
    {
      uploadUrl: 'https://storage.googleapis.com/matcha-demo/match2.mp4',
      status: MatchStatus.COMPLETED,
      duration: 3600, // 60 mins
      summary: 'Dominant 3-0 performance by the home team.',
      topSpeedKmh: 28.1,
      progress: 100,
      userId: user.id,
    },
    {
      uploadUrl: 'https://storage.googleapis.com/matcha-demo/match3.mp4',
      status: MatchStatus.PROCESSING,
      duration: 0,
      summary: 'Processing AI analytics...',
      progress: 45,
      userId: user.id,
    },
  ];

  for (const data of matchesData) {
    const match = await prisma.match.create({
      data: {
        ...data,
        events: data.status === MatchStatus.COMPLETED ? {
          create: [
            { type: EventType.GOAL, timestamp: 1200, confidence: 0.98, commentary: 'Amazing long-range strike!' },
            { type: EventType.SAVE, timestamp: 2500, confidence: 0.92, commentary: 'Stunning fingertip save.' },
            { type: EventType.YELLOW_CARD, timestamp: 3100, confidence: 0.85, commentary: 'Late challenge in midfield.' },
          ]
        } : undefined,
        highlights: data.status === MatchStatus.COMPLETED ? {
          create: [
            { startTime: 1190, endTime: 1210, score: 0.95, eventType: 'GOAL', commentary: 'Goal Highlight' },
            { startTime: 2490, endTime: 2510, score: 0.88, eventType: 'SAVE', commentary: 'Save Highlight' },
          ]
        } : undefined,
        emotionScores: data.status === MatchStatus.COMPLETED ? {
          create: [
            { timestamp: 1200, audioScore: 0.9, motionScore: 0.8, contextWeight: 1.0, finalScore: 0.85 },
            { timestamp: 2500, audioScore: 0.7, motionScore: 0.9, contextWeight: 1.0, finalScore: 0.8 },
          ]
        } : undefined,
      },
    });
    console.log(`✅ Created match: ${match.id} (${match.status})`);
  }

  console.log('🚀 Seeding completed successfully!');
}

main()
  .catch((e) => {
    console.error('❌ Seeding failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
