import ffmpeg from 'fluent-ffmpeg';
import ffmpegStatic from 'ffmpeg-static';
import path from 'path';
import fs from 'fs';

if (ffmpegStatic) {
    ffmpeg.setFfmpegPath(ffmpegStatic);
}

interface FrameExtractionOptions {
    videoPath: string;
    outputDir: string;
    fps?: number; // e.g., 1 for 1 frame/sec, 0.5 for 1 frame every 2 sec
}

export async function extractFrames({
    videoPath,
    outputDir,
    fps = 1,
}: FrameExtractionOptions): Promise<string[]> {
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    return new Promise((resolve, reject) => {
        ffmpeg(videoPath)
            .outputOptions([`-vf fps=${fps}`, '-qscale:v 2']) // High JPEG quality
            .output(path.join(outputDir, 'frame_%04d.jpg'))
            .on('end', () => {
                const files = fs
                    .readdirSync(outputDir)
                    .filter((f) => f.endsWith('.jpg'))
                    .map((f) => path.join(outputDir, f));
                resolve(files);
            })
            .on('error', (err) => reject(err))
            .run();
    });
}