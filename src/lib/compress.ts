/**
 * Client-side video compression orchestrator.
 *
 * Spawns `compress-worker.ts` as a Web Worker so the main thread stays
 * responsive.  Returns a promise that resolves with the compressed Blob
 * once encoding completes.
 *
 * Usage:
 *   const result = await compressVideo({
 *     file, duration, width, height, fps,
 *     onProgress: (pct) => setProgress(pct),
 *   })
 *   // result.blob is the compressed MP4
 */

import type { CompressionProfile } from './compress-profiles'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CompressOptions {
  /** The original video File object */
  file: File
  /** Duration in seconds (from video metadata) */
  duration: number
  /** Source width in pixels */
  width: number
  /** Source height in pixels */
  height: number
  /** Source frame-rate (fps) */
  fps: number
  /** Optional progress callback (0–100) */
  onProgress?: (percent: number) => void
}

export interface CompressResult {
  /** The compressed video Blob */
  blob: Blob
  /** Suggested output filename */
  filename: string
  /** The profile that was used */
  profile: CompressionProfile
  /** Original file size in bytes */
  originalSize: number
  /** Compressed file size in bytes */
  compressedSize: number
  /** Percentage reduction (0–100) */
  reduction: number
}

// ---------------------------------------------------------------------------
// Worker management
// ---------------------------------------------------------------------------

let activeWorker: Worker | null = null

/**
 * Compress a video file using FFmpeg.wasm inside a Web Worker.
 *
 * The worker is created on-demand and terminated after each compression
 * so WASM memory is released between calls.
 */
export function compressVideo(options: CompressOptions): Promise<CompressResult> {
  // Cancel any in-progress compression
  if (activeWorker) {
    activeWorker.terminate()
    activeWorker = null
  }

  return new Promise<CompressResult>((resolve, reject) => {
    const worker = new Worker(
      new URL('./compress-worker.ts', import.meta.url),
      { type: 'module' },
    )
    activeWorker = worker

    worker.onmessage = (e: MessageEvent) => {
      const data = e.data as
        | { type: 'progress'; progress: number }
        | { type: 'loaded' }
        | {
            type: 'done'
            blob: Blob
            filename: string
            profile: CompressionProfile
            originalSize: number
            compressedSize: number
          }
        | { type: 'error'; error: string }

      switch (data.type) {
        case 'progress':
          options.onProgress?.(data.progress)
          break

        case 'loaded':
          // FFmpeg.wasm core is loaded — still compressing
          break

        case 'done': {
          activeWorker = null
          worker.terminate()
          resolve({
            blob: data.blob,
            filename: data.filename,
            profile: data.profile,
            originalSize: data.originalSize,
            compressedSize: data.compressedSize,
            reduction: Math.round(
              (1 - data.compressedSize / data.originalSize) * 100,
            ),
          })
          break
        }

        case 'error':
          activeWorker = null
          worker.terminate()
          reject(new Error(data.error))
          break
      }
    }

    worker.onerror = (err) => {
      activeWorker = null
      worker.terminate()
      reject(new Error(err.message || 'FFmpeg Worker failed'))
    }

    // Transfer the file buffer to the worker (zero-copy)
    options.file.arrayBuffer().then((buffer) => {
      worker.postMessage(
        {
          fileBuffer: buffer,
          filename: options.file.name,
          duration: options.duration,
          width: options.width,
          height: options.height,
          fps: options.fps,
        },
        [buffer],
      )
    })
  })
}

/**
 * Cancel an in-progress compression (terminates the worker).
 */
export function cancelCompression(): void {
  if (activeWorker) {
    activeWorker.terminate()
    activeWorker = null
  }
}
