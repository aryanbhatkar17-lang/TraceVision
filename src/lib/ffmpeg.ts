/**
 * ffmpeg.ts — DEPRECATED / STUB
 * ==============================
 * This file is intentionally empty. FFmpeg loading is now handled
 * entirely inside compress-worker.ts via importScripts() from the UMD
 * CDN, which bypasses Webpack's module resolver.
 *
 * DO NOT re-add any @ffmpeg/ffmpeg or @ffmpeg/util imports here.
 * Importing those packages on the main thread causes Webpack to
 * intercept their internal dynamic import(blobURL) calls at runtime,
 * crashing with:
 *   "Error: Cannot find module 'blob:http://localhost:3000/...'"
 */

export {};