
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  const match = await prisma.match.findUnique({
    where: { id: '2085826f-69e3-456c-9ac4-b4780a8dfacb' },
    include: { highlights: true }
  });
  console.log(JSON.stringify(match, null, 2));
}

main().finally(() => prisma.$disconnect());
