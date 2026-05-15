-- CreateEnum
CREATE TYPE "MatchStatus" AS ENUM ('UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "EventType" AS ENUM ('GOAL', 'FOUL', 'TACKLE', 'SAVE', 'CELEBRATION', 'HIGHLIGHT', 'PENALTY', 'RED_CARD', 'YELLOW_CARD', 'CORNER', 'OFFSIDE');

-- CreateTable
CREATE TABLE "Match" (
    "id" TEXT NOT NULL,
    "uploadUrl" TEXT NOT NULL,
    "status" "MatchStatus" NOT NULL DEFAULT 'UPLOADED',
    "duration" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "summary" TEXT,
    "trackingData" JSONB,
    "teamColors" JSONB,
    "progress" INTEGER NOT NULL DEFAULT 0,
    "highlightReelUrl" TEXT,
    "highlightReelPortraitUrl" TEXT,
    "heatmapUrl" TEXT,
    "thumbnailUrl" TEXT,
    "topSpeedKmh" DOUBLE PRECISION,
    "userId" TEXT,

    CONSTRAINT "Match_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Event" (
    "id" TEXT NOT NULL,
    "matchId" TEXT NOT NULL,
    "timestamp" DOUBLE PRECISION NOT NULL,
    "type" "EventType" NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finalScore" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "commentary" TEXT,

    CONSTRAINT "Event_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "EmotionScore" (
    "id" TEXT NOT NULL,
    "matchId" TEXT NOT NULL,
    "timestamp" DOUBLE PRECISION NOT NULL,
    "audioScore" DOUBLE PRECISION NOT NULL,
    "motionScore" DOUBLE PRECISION NOT NULL,
    "contextWeight" DOUBLE PRECISION NOT NULL,
    "finalScore" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "EmotionScore_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Highlight" (
    "id" TEXT NOT NULL,
    "matchId" TEXT NOT NULL,
    "startTime" DOUBLE PRECISION NOT NULL,
    "endTime" DOUBLE PRECISION NOT NULL,
    "score" DOUBLE PRECISION NOT NULL,
    "eventType" TEXT,
    "commentary" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "videoUrl" TEXT,

    CONSTRAINT "Highlight_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "firstName" TEXT,
    "lastName" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Event_matchId_idx" ON "Event"("matchId");

-- CreateIndex
CREATE INDEX "Event_timestamp_idx" ON "Event"("timestamp");

-- CreateIndex
CREATE INDEX "EmotionScore_matchId_idx" ON "EmotionScore"("matchId");

-- CreateIndex
CREATE INDEX "EmotionScore_timestamp_idx" ON "EmotionScore"("timestamp");

-- CreateIndex
CREATE INDEX "Highlight_matchId_idx" ON "Highlight"("matchId");

-- CreateIndex
CREATE INDEX "Highlight_startTime_idx" ON "Highlight"("startTime");

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- AddForeignKey
ALTER TABLE "Match" ADD CONSTRAINT "Match_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Event" ADD CONSTRAINT "Event_matchId_fkey" FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "EmotionScore" ADD CONSTRAINT "EmotionScore_matchId_fkey" FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Highlight" ADD CONSTRAINT "Highlight_matchId_fkey" FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE CASCADE ON UPDATE CASCADE;
