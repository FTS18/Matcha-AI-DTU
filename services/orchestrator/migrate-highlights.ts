
import { PrismaClient } from '@prisma/client';
import * as fs from 'fs';
import * as path from 'path';

const prisma = new PrismaClient();
const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');

async function main() {
  console.log(`Uploads dir: ${uploadsDir}`);
  const highlights = await prisma.highlight.findMany();
  console.log(`Checking ${highlights.length} highlights...`);
  
  for (const h of highlights) {
    if (!h.videoUrl) continue;
    
    // Handle cases where videoUrl might be absolute
    let fileName = h.videoUrl;
    if (fileName.includes('/uploads/')) {
        fileName = fileName.split('/uploads/').pop()!;
    } else {
        fileName = path.basename(fileName);
    }
    
    const currentPath = path.join(uploadsDir, fileName);
    
    if (!fs.existsSync(currentPath)) {
      console.log(`[MISSING] ${fileName} (Highlight ID: ${h.id})`);
      
      // Try to find alternative prefix
      if (fileName.startsWith('clip_')) {
        const h264Name = fileName.replace('clip_', 'sa_h264_');
        const h264Path = path.join(uploadsDir, h264Name);
        
        if (fs.existsSync(h264Path)) {
          console.log(`  -> Found alternative: ${h264Name}. Updating DB...`);
          await prisma.highlight.update({
            where: { id: h.id },
            data: { videoUrl: `/uploads/${h264Name}` }
          });
        } else {
           // Maybe it's missing the match ID part or has extra ar_tag?
           // The pattern in video_utils was f"clip_{match_id}_{i}{ar_tag}.mp4"
           // And in sa was f"sa_h264_{match_id}_{clip_idx}.mp4"
           // They should match exactly if we just swap the prefix.
           console.log(`  -> Not found at ${h264Path} either.`);
        }
      }
    } else {
        // console.log(`[OK] ${fileName}`);
    }
  }
  
  // Also check Match.uploadUrl for absolute vs relative
  const matches = await prisma.match.findMany();
  for (const m of matches) {
      if (m.uploadUrl && m.uploadUrl.startsWith('http')) {
          const fileName = m.uploadUrl.split('/uploads/').pop()!;
          console.log(`[MATCH] Updating absolute URL for match ${m.id} -> /uploads/${fileName}`);
          await prisma.match.update({
              where: { id: m.id },
              data: { uploadUrl: `/uploads/${fileName}` }
          });
      }
  }
}

main().catch(err => console.error(err)).finally(() => prisma.$disconnect());
