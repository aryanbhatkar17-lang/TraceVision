export interface CompressionProfile {
  label: string;
  crf: number;
  maxFps: number;
  maxResolution: string;
  preset: string;
}

export const COMPRESSION_PROFILES: Record<string, CompressionProfile> = {
  fast: {
    label: 'Fast',
    crf: 32,
    maxFps: 15,
    maxResolution: '854:480',
    preset: 'ultrafast',
  },
  balanced: {
    label: 'Balanced',
    crf: 28,
    maxFps: 24,
    maxResolution: '1280:720',
    preset: 'veryfast',
  },
  high: {
    label: 'High Quality',
    crf: 24,
    maxFps: 30,
    maxResolution: '1920:1080',
    preset: 'fast',
  }
};
