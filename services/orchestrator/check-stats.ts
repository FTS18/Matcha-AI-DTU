
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  const matches = await prisma.match.count({ where: { status: 'COMPLETED' } });
  const events = await prisma.event.count();
  const highlights = await prisma.highlight.count();
  const durationSum = await prisma.match.aggregate({
    _sum: { duration: true },
    where: { status: 'COMPLETED' }
  });
  
  console.log({
    matches,
    events,
    highlights,
    totalDuration: durationSum._sum.duration || 0
  });
}

main().finally(() => prisma.$disconnect());
