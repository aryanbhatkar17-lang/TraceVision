/**
 * Web Worker for client-side video compression via FFmpeg.wasm.
 *
 * Runs entirely off the main thread so the UI stays responsive during
 * the (potentially slow) transcode.  Communication is via postMessage:
 *
 *   IN:  { fileBuffer, filename, duration, width, height, fps }
 *   OUT: { type: 'progress', progress: number }
 *        { type: 'loaded' }
 *        { type: 'done', blob, filename, profile, originalSize, compressedSize }
 *        { type: 'error', error: string }
 */

import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util'
import { resolveProfile } from './compress-profiles'

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async (e: MessageEvent) => {
  const { fileBuffer, filename, duration, width, height, fps } = e.data as {
    fileBuffer: ArrayBuffer
    filename: string
    duration: number
    width: number
    height: number
    fps: number
  }

  try {
    // 1. Resolve encoding profile
    const profile = resolveProfile(duration, width, height, fps)

    // 2. Load FFmpeg.wasm
    const ffmpeg = new FFmpeg()
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'

    await ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    })

    self.postMessage({ type: 'loaded' })

    // 3. Progress reporting
    ffmpeg.on('progress', ({ progress }) => {
      self.postMessage({
        type: 'progress',
        progress: Math.round(progress * 100),
      })
    })

    // 4. Write input to virtual FS
    const inputName = 'input.mp4'
    const outputName = 'output.mp4'
    await ffmpeg.writeFile(inputName, new Uint8Array(fileBuffer))

    // 5. Build FFmpeg arguments
    const [maxW, maxH] = profile.maxResolution.split(':')
    const vf = [
      `scale='min(${maxW},iw)':'min(${maxH},ih)':force_original_aspect_ratio=decrease`,
      `fps=${profile.maxFps}`,
    ].join(',')

    const args: string[] = [
      '-i', inputName,
      '-vf', vf,
      '-c:v', 'libx264',
      '-crf', String(profile.crf),
      '-preset', profile.preset,
      '-movflags', '+faststart',
    ]

    if (profile.stripAudio) {
      args.push('-an')
    }

    args.push(outputName)

    // 6. Execute
    await ffmpeg.exec(args)

    // 7. Read output and build blob
    const outputData = await ffmpeg.readFile(outputName)
    // Convert to Uint8Array then copy to a fresh ArrayBuffer to avoid SharedArrayBuffer type issues
    const outputBytes = outputData instanceof Uint8Array ? outputData : new TextEncoder().encode(outputData)
    const ab = new ArrayBuffer(outputBytes.byteLength)
    new Uint8Array(ab).set(outputBytes)
    const blob = new Blob([ab], { type: 'video/mp4' })

    // 8. Cleanup virtual FS
    await ffmpeg.deleteFile(inputName)
    await ffmpeg.deleteFile(outputName)

    // 9. Terminate ffmpeg to free WASM memory
    ffmpeg.terminate()

    // 10. Send result back
    self.postMessage({
      type: 'done',
      blob,
      filename: filename.replace(/\.[^.]+$/, '_compressed.mp4'),
      profile,
      originalSize: fileBuffer.byteLength,
      compressedSize: blob.size,
    })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    self.postMessage({ type: 'error', error: message })
  }
}
