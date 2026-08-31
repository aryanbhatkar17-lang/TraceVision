import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';
import { COMPRESSION_PROFILES } from './compress-profiles';

export interface CompressOptions {
  file: File;
  /** Optional video metadata — accepted for API compatibility but the worker
   *  uses profile-based settings (CRF/resolution) rather than these values. */
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
  profileKey?: 'fast' | 'balanced' | 'high';
  onProgress?: (progress: number) => void;
}

export interface CompressResult {
  blob: Blob;
  filename: string;
  originalSize: number;
  compressedSize: number;
  reduction: number;
}

// Keep a singleton instance so we don't reload the WASM core on every compression
let ffmpegInstance: FFmpeg | null = null;

const loadFFmpeg = async (): Promise<FFmpeg> => {
  if (ffmpegInstance) return ffmpegInstance;

  const ffmpeg = new FFmpeg();

  // Static URLs to avoid Webpack "too dynamic" errors
  const coreJsURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js';
  const coreWasmURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.wasm';

  await ffmpeg.load({
    coreURL: await toBlobURL(coreJsURL, 'text/javascript'),
    wasmURL: await toBlobURL(coreWasmURL, 'application/wasm'),
  });

  ffmpegInstance = ffmpeg;
  return ffmpeg;
};

export const compressVideo = async (options: CompressOptions): Promise<CompressResult> => {
  const { file, profileKey, onProgress } = options;
  
  const ffmpeg = await loadFFmpeg();
  
  // Set up progress listener
  const progressHandler = ({ progress }: { progress: number }) => {
    if (onProgress) {
      onProgress(Math.round(progress * 100));
    }
  };
  
  ffmpeg.on('progress', progressHandler);

  const inputName = 'input.mp4';
  const outputName = 'output.mp4';
  
  try {
    const profile = COMPRESSION_PROFILES[profileKey ?? 'balanced'] || COMPRESSION_PROFILES.balanced;
    
    // Write the file to FFmpeg's in-memory filesystem
    const fileData = await fetchFile(file);
    await ffmpeg.writeFile(inputName, fileData);

    const [maxW, maxH] = profile.maxResolution.split(':');
    const vf = `scale='min(${maxW},iw)':'min(${maxH},ih)':force_original_aspect_ratio=decrease,fps=${profile.maxFps}`;

    // Execute compression
    // FFmpeg.wasm automatically spawns a background Web Worker internally,
    // so this await will NOT freeze the React main thread.
    await ffmpeg.exec([
      '-i', inputName,
      '-vf', vf,
      '-c:v', 'libx264',
      '-crf', String(profile.crf),
      '-preset', profile.preset,
      '-movflags', '+faststart',
      outputName
    ]);

    // Read the result
    const outputData = await ffmpeg.readFile(outputName);
    
    // Convert to Blob
    const outputBytes = outputData instanceof Uint8Array ? outputData : new TextEncoder().encode(outputData as string);
    const ab = new ArrayBuffer(outputBytes.byteLength);
    new Uint8Array(ab).set(outputBytes);
    const blob = new Blob([ab], { type: 'video/mp4' });

    // Cleanup memory
    await ffmpeg.deleteFile(inputName);
    await ffmpeg.deleteFile(outputName);
    
    return {
      blob,
      filename: file.name.replace(/\.[^.]+$/, '_compressed.mp4'),
      originalSize: file.size,
      compressedSize: blob.size,
      reduction: Math.round((1 - blob.size / file.size) * 100),
    };
  } catch (error) {
    throw error;
  } finally {
    // Remove the listener so it doesn't fire for future compressions
    ffmpeg.off('progress', progressHandler);
  }
};
