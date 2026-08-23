/**
 * Adaptive compression profile resolver.
 *
 * Selects FFmpeg encoding parameters based on the source video's duration,
 * resolution, and frame-rate so we upload the smallest file that still
 * carries enough visual detail for the Gemini Vision model (which only
 * ever sees 640×360 JPEG frames downstream).
 */

export interface CompressionProfile {
  /** Human-readable label for logs / UI */
  label: string
  /** x264 CRF value – lower = higher quality (18–51) */
  crf: number
  /** Maximum output frame-rate (fps) */
  maxFps: number
  /** Max resolution as "W:H" — aspect-ratio is preserved */
  maxResolution: string
  /** x264 encode speed preset */
  preset: 'ultrafast' | 'fast' | 'medium'
  /** Whether to strip the audio track entirely */
  stripAudio: boolean
}

/**
 * Resolve the best compression profile for the given source metadata.
 *
 * The heuristic works because:
 * - CCTV footage is almost always re-encoded at 640×360 for the vision model,
 *   so there is *zero* perceptual loss from down-scaling before upload.
 * - Reducing fps from 30→10 or 5 has negligible impact when the extractor
 *   only samples at 1 fps anyway.
 * - CRF 28–35 produces output that looks identical at 640×360.
 */
export function resolveProfile(
  durationSec: number,
  width: number,
  height: number,
  _fps: number,
): CompressionProfile {
  // Very short clips — keep quality high, minimal downsampling
  if (durationSec < 60) {
    return {
      label: 'quick',
      crf: 28,
      maxFps: 15,
      maxResolution: '1280:720',
      preset: 'fast',
      stripAudio: true,
    }
  }

  // Medium clips — balanced trade-off
  if (durationSec < 300) {
    return {
      label: 'balanced',
      crf: 30,
      maxFps: 10,
      maxResolution: '640:360',
      preset: 'ultrafast',
      stripAudio: true,
    }
  }

  // Long clips — aggressive compression
  if (durationSec < 600) {
    return {
      label: 'compact',
      crf: 32,
      maxFps: 8,
      maxResolution: '480:270',
      preset: 'ultrafast',
      stripAudio: true,
    }
  }

  // Very long CCTV archives — maximum compression
  return {
    label: 'archive',
    crf: 35,
    maxFps: 5,
    maxResolution: '320:180',
    preset: 'ultrafast',
    stripAudio: true,
  }
}
