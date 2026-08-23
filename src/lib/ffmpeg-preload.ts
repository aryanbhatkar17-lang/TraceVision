/**
 * Background preloader for FFmpeg.wasm.
 *
 * The WASM core is ~25 MB and takes a few seconds to download + compile.
 * By calling `preloadFFmpeg()` when the Dashboard mounts, the core will
 * be ready before the user clicks "Analyze" — shaving seconds off the
 * first compression.
 *
 * This module does NOT hold a persistent FFmpeg instance.  Each
 * compression still creates its own worker/instance so memory is
 * released after each use.  The preload only ensures the browser
 * caches the WASM binary after the first fetch.
 */

let preloadPromise: Promise<void> | null = null

/**
 * Kick off a background download + compile of the FFmpeg.wasm core.
 * Safe to call multiple times — subsequent calls are no-ops.
 */
export function preloadFFmpeg(): void {
  if (preloadPromise) return

  preloadPromise = (async () => {
    const { FFmpeg } = await import('@ffmpeg/ffmpeg')
    const { toBlobURL } = await import('@ffmpeg/util')

    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'
    const ffmpeg = new FFmpeg()

    await ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    })

    // Terminate immediately — we only needed the browser to cache the files
    ffmpeg.terminate()

    if (process.env.NODE_ENV !== 'production') {
      console.log('[FFmpeg] WASM core preloaded and cached')
    }
  })().catch((err) => {
    console.warn('[FFmpeg] Preload failed (will retry on first use):', err)
    preloadPromise = null // Allow retry
  })
}

/**
 * Returns true if the preload has been initiated (not necessarily complete).
 */
export function isPreloading(): boolean {
  return preloadPromise !== null
}
