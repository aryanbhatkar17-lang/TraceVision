import sharp from 'sharp'
import pixelmatch from 'pixelmatch'

/**
 * CONFIGURATION CONSTANTS:
 * 
 * 1. THUMB_SIZE = 64
 *    We shrink 1080p/4K frames down to a tiny 64x64 pixel grid (4,096 total pixels).
 *    Comparing a 4,096-pixel buffer takes < 1 millisecond on CPU, making it near-instant.
 * 
 * 2. DEFAULT_MOTION_THRESHOLD = 0.03
 *    0.03 = 3% threshold.
 *    If fewer than 3% of pixels changed between Frame A and Frame B, the scene is static (empty hallway/road).
 *    If 3% or more changed, active movement occurred (person/car entered).
 */
const THUMB_SIZE = 64
const DEFAULT_MOTION_THRESHOLD = 0.03

/**
 * Interface representing a frame that passed the motion gate
 */
export interface FilteredFrame {
    path: string          // Local file path on disk to the extracted JPEG
    second: number        // Original real-world second in the video (e.g., 14s)
    motionDelta: number   // The percentage of pixels changed (e.g., 0.12 = 12% motion)
}

/**
 * Helper: Normalizes an image frame into a small 64x64 raw RGBA byte buffer.
 * 
 * Why this is needed:
 * - 'pixelmatch' requires raw uncompressed pixel data (RGBA channels).
 * - 'sharp' resizes and extracts the raw byte buffer directly in Node.js memory.
 * 
 * @param imagePath - Path to the JPEG frame on disk
 * @returns Buffer containing raw pixel bytes (64 * 64 * 4 = 16,384 bytes)
 */
async function getFrameBuffer(imagePath: string): Promise<Buffer> {
    return await sharp(imagePath)
        .resize(THUMB_SIZE, THUMB_SIZE, { fit: 'fill' }) // Scale down to 64x64
        .ensureAlpha()                                   // Ensure 4 channels (Red, Green, Blue, Alpha)
        .raw()                                           // Strip JPEG headers; return pure byte array
        .toBuffer()
}

/**
 * Computes the perceptual difference percentage between two sequential frames.
 * 
 * @param prevPath - File path of the previous reference frame
 * @param currPath - File path of the current frame to test
 * @returns A decimal number between 0.0 (identical) and 1.0 (completely different)
 */
export async function calculateMotionDelta(
    prevPath: string,
    currPath: string
): Promise<number> {
    // 1. Convert both frames to 64x64 raw buffers concurrently
    const [prevBuf, currBuf] = await Promise.all([
        getFrameBuffer(prevPath),
        getFrameBuffer(currPath),
    ])

    const totalPixels = THUMB_SIZE * THUMB_SIZE // 4,096 pixels

    // 2. Run pixelmatch algorithm
    // threshold: 0.15 ignores tiny camera sensor noise / subtle compression artifacts
    const diffPixels = pixelmatch(
        prevBuf,
        currBuf,
        undefined,             // Passing undefined because we don't need a visual diff image output
        THUMB_SIZE,
        THUMB_SIZE,
        { threshold: 0.15 }
    )

    // 3. Return the ratio of changed pixels (e.g., 400 / 4096 = ~0.097 -> 9.7% changed)
    return diffPixels / totalPixels
}

/**
 * Main Filtering Engine:
 * Iterates through all extracted frames from FFmpeg and filters out redundant static footage.
 * 
 * How it keeps timestamps accurate:
 * - When frame 10 (at 10s) is retained, and frames 11-20 are dropped because nothing moved,
 *   the next motion frame at frame 21 will still record its second as 21s!
 * 
 * @param framePaths - Array of file paths to the extracted JPEG frames (sorted chronologically)
 * @param stepSeconds - The sample rate FFmpeg used (default: 1 frame per 1 second)
 * @param motionThreshold - Minimum percentage of movement needed to retain the frame (default: 0.03)
 */
export async function filterStaticFrames(
    framePaths: string[],
    stepSeconds = 1,
    motionThreshold = DEFAULT_MOTION_THRESHOLD
): Promise<FilteredFrame[]> {
    // Guard clause: Return empty list if no frames exist
    if (framePaths.length === 0) return []

    const activeFrames: FilteredFrame[] = []

    // Step A: Always keep the very first frame (0s) as the initial visual reference baseline
    activeFrames.push({
        path: framePaths[0],
        second: 0,
        motionDelta: 1.0, // 1.0 represents 100% (initial baseline)
    })

    let lastRetainedIndex = 0

    // Step B: Compare each subsequent frame against the last retained motion frame
    for (let i = 1; i < framePaths.length; i++) {
        const delta = await calculateMotionDelta(
            framePaths[lastRetainedIndex], // Compare against the last frame where movement was kept
            framePaths[i]                 // Current frame being inspected
        )

        // Step C: If pixel change exceeds threshold, store the frame and update the baseline pointer
        if (delta >= motionThreshold) {
            activeFrames.push({
                path: framePaths[i],
                second: i * stepSeconds,                 // Preserves original video timestamp
                motionDelta: Number(delta.toFixed(3)),
            })
            lastRetainedIndex = i                      // Advance reference index to the new active frame
        }
        // If delta < motionThreshold, loop continues and the static frame is skipped (dropped)
    }

    return activeFrames
}