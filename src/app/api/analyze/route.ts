import { NextRequest, NextResponse } from 'next/server'
import { AuditMatch, AuditResponse } from '@/types/audit'
import { extractFrames } from '@/lib/ffmpeg'
import { filterStaticFrames } from '@/lib/motion-filter'
import path from 'path'
import fs from 'fs/promises'
import os from 'os'

// Next.js Route Segment Configuration:
// 1. Force dynamic execution for every audit request (no caching)
// 2. Extend timeout window to 300 seconds (5 minutes) for processing video files
export const dynamic = 'force-dynamic'
export const maxDuration = 300

/**
 * Utility: Converts numeric seconds into standardized MM:SS or HH:MM:SS format
 * Example: 75 -> "01:15", 3665 -> "01:01:05"
 */
function formatSeconds(sec: number): string {
  const hrs = Math.floor(sec / 3600)
  const mins = Math.floor((sec % 3600) / 60)
  const secs = Math.floor(sec % 60)
  if (hrs > 0) {
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

export async function POST(req: NextRequest) {
  let tempDir = '' // Tracked for reliable disk cleanup in the finally block

  try {
    // -------------------------------------------------------------
    // STEP 1: Parse Incoming Multipart Form Data from the Frontend
    // -------------------------------------------------------------
    const formData = await req.formData()
    const videoFile = (formData.get('video') || formData.get('file')) as File | null
    const query = (formData.get('query') as string) || ''
    const duration = parseFloat((formData.get('duration') as string) || '0')

    // Validate that a video file was actually provided
    if (!videoFile || typeof videoFile === 'string') {
      return NextResponse.json(
        { error: 'No video file provided for analysis.' },
        { status: 400 }
      )
    }

    // Validate Gemini API Key existence
    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: 'GEMINI_API_KEY is not set in .env.local' },
        { status: 500 }
      )
    }

    // -------------------------------------------------------------
    // STEP 2: Save Uploaded Video to OS Temporary Directory
    // -------------------------------------------------------------
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'cctv-audit-'))
    const videoFilePath = path.join(tempDir, 'input_feed.mp4')
    const framesOutputDir = path.join(tempDir, 'frames')

    // Convert browser File object into a Node.js Buffer and write to local temp disk
    const fileBuffer = Buffer.from(await videoFile.arrayBuffer())
    await fs.writeFile(videoFilePath, fileBuffer)

    // -------------------------------------------------------------
    // STEP 3: Extract 1 Frame Per Second Using FFmpeg
    // -------------------------------------------------------------
    const rawFramePaths = await extractFrames({
      videoPath: videoFilePath,
      outputDir: framesOutputDir,
      fps: 1, // Samples 1 frame every 1 second
    })

    if (rawFramePaths.length === 0) {
      throw new Error('FFmpeg was unable to extract frames from the uploaded video.')
    }

    // Sort frames chronologically (e.g. frame_0001.jpg, frame_0002.jpg)
    rawFramePaths.sort()

    // -------------------------------------------------------------
    // STEP 4: Motion-Delta Filtering via Sharp + Pixelmatch
    // -------------------------------------------------------------
    // Drops static non-moving frames (< 3% pixel change) to save 60-80% of tokens & processing time
    const activeFrames = await filterStaticFrames(rawFramePaths, 1, 0.03)

    // Fallback: If camera scene is completely still, keep at least the initial reference frame
    const framesToAnalyze =
      activeFrames.length > 0
        ? activeFrames
        : [{ path: rawFramePaths[0], second: 0, motionDelta: 0 }]

    const droppedPercentage = (
      ((rawFramePaths.length - framesToAnalyze.length) / rawFramePaths.length) *
      100
    ).toFixed(1)

    console.log(
      `[Motion-Delta] Raw Frames: ${rawFramePaths.length} -> Active Motion Frames: ${framesToAnalyze.length} ` +
      `(${droppedPercentage}% static non-moving frames discarded)`
    )

    // -------------------------------------------------------------
    // STEP 5: Sample & Encode Active Frames to Base64 for Gemini
    // -------------------------------------------------------------
    // Cap at a max of 60 frames per API payload to stay well within Gemini Flash token windows
    const maxFrames = 60
    const step = Math.max(1, Math.floor(framesToAnalyze.length / maxFrames))
    const sampledFrames = framesToAnalyze.filter((_, idx) => idx % step === 0)

    // Convert each retained JPEG frame into Base64 inline data
    const imageParts = await Promise.all(
      sampledFrames.map(async (frame) => {
        const frameBuffer = await fs.readFile(frame.path)
        return {
          inlineData: {
            data: frameBuffer.toString('base64'),
            mimeType: 'image/jpeg',
          },
        }
      })
    )

    // Build an explicit timestamp index so Gemini maps each frame back to real video seconds
    const timestampsIndex = sampledFrames
      .map((f, idx) => `Frame ${idx}: ${f.second}s`)
      .join(', ')

    // -------------------------------------------------------------
    // STEP 6: Define Strict JSON Schema for Structured Incident Output
    // -------------------------------------------------------------
    const auditSchema = {
      type: 'OBJECT',
      properties: {
        matches: {
          type: 'ARRAY',
          items: {
            type: 'OBJECT',
            properties: {
              start_seconds: {
                type: 'NUMBER',
                description: 'Exact starting timestamp second where the query target appears in the video.',
              },
              end_seconds: {
                type: 'NUMBER',
                description: 'Exact ending timestamp second where the query target leaves the frame.',
              },
              category: {
                type: 'STRING',
                description: 'PERSON, VEHICLE, OBJECT, SECURITY, or ANOMALY',
              },
              description: {
                type: 'STRING',
                description: 'Concise description of the observed target activity.',
              },
              confidence: {
                type: 'NUMBER',
                description: 'Confidence score between 0.0 and 1.0 based strictly on visual clarity.',
              },
            },
            required: ['start_seconds', 'end_seconds', 'category', 'description', 'confidence'],
          },
        },
      },
      required: ['matches'],
    }

    // -------------------------------------------------------------
    // STEP 7: Run Multimodal Vision Inference (Gemini 3.6 Flash)
    // -------------------------------------------------------------
    const geminiResponse = await fetch(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key=' +
        encodeURIComponent(process.env.GEMINI_API_KEY),
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                ...imageParts.map((part) => ({ inline_data: part.inlineData })),
                { text: `You are an automated CCTV Video Surveillance Audit Engine.
The provided sequence contains ONLY frames where active visual motion was detected.
Frame Timestamp Index: [ ${timestampsIndex} ]

Target Audit Query: "${query}"

Strict Instructions:
1. Examine each frame carefully for the presence of the query target.
2. If the target is NOT visible in the frames, return an empty array for matches.
3. Map start_seconds and end_seconds directly to the real timestamps provided in the Timestamp Index.
4. Output valid JSON adhering strictly to the response schema.`,
                },
              ],
            },
          ],
          generationConfig: {
            responseMimeType: 'application/json',
            responseSchema: auditSchema,
          },
        }),
      }
    )

    if (!geminiResponse.ok) {
      throw new Error(`Gemini API request failed (${geminiResponse.status})`)
    }

    const response = (await geminiResponse.json()) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>
    }
    const responseText = response.candidates?.[0]?.content?.parts?.[0]?.text

    // Parse the structured JSON response
    const parsed = JSON.parse(responseText || '{"matches":[]}')
    const rawMatches = parsed.matches || []

    // -------------------------------------------------------------
    // STEP 8: Format Matches for the Frontend Dashboard & Timeline
    // -------------------------------------------------------------
    const formattedMatches: AuditMatch[] = rawMatches.map((m: any, index: number) => ({
      id: `match-${index + 1}`,
      start_time: formatSeconds(m.start_seconds),
      end_time: formatSeconds(m.end_seconds),
      start_seconds: Math.round(m.start_seconds),
      end_seconds: Math.round(m.end_seconds),
      category: m.category || 'PERSON',
      description: m.description,
      confidence: Number(m.confidence?.toFixed(2)) || 0.9,
      chunk_id: `chunk_${String(index + 1).padStart(3, '0')}`,
    }))

    const responsePayload: AuditResponse = {
      matches: formattedMatches,
      total_chunks: 1,
      video_duration: duration || rawFramePaths.length,
      query,
    }

    // Return the formatted audit result back to the frontend
    return NextResponse.json(responsePayload)
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error during analysis'
    console.error('API /api/analyze error:', err)
    return NextResponse.json({ error: message }, { status: 500 })
  } finally {
    // -------------------------------------------------------------
    // STEP 9: Clean Up Temporary Disk Files
    // -------------------------------------------------------------
    if (tempDir) {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {})
    }
  }
}