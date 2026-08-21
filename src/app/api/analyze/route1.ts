import { NextRequest, NextResponse } from 'next/server'
import { AuditMatch, AuditResponse } from '@/types/audit'
import { extractFrames } from '@/lib/ffmpeg'

type Schema = {
  type: string
  properties?: Record<string, Schema>
  items?: Schema
  required?: string[]
  description?: string
}

type RawMatch = {
  start_seconds: number
  end_seconds: number
  category?: string
  description: string
  confidence?: number
}

import path from 'path'
import fs from 'fs/promises'
import os from 'os'

export const dynamic = 'force-dynamic'
export const maxDuration = 300 // 5-minute timeout window for processing

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
  let tempDir = ''

  try {
    const formData = await req.formData()
    const videoFile = (formData.get('video') || formData.get('file')) as File | null
    const query = (formData.get('query') as string) || ''
    const duration = parseFloat((formData.get('duration') as string) || '0')

    if (!videoFile || typeof videoFile === 'string') {
      return NextResponse.json(
        { error: 'No video file provided for analysis.' },
        { status: 400 }
      )
    }

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: 'GEMINI_API_KEY is not set in .env.local' },
        { status: 500 }
      )
    }

    // 1. Create temporary directory and save video
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'cctv-audit-'))
    const videoFilePath = path.join(tempDir, 'input_feed.mp4')
    const framesOutputDir = path.join(tempDir, 'frames')

    const fileBuffer = Buffer.from(await videoFile.arrayBuffer())
    await fs.writeFile(videoFilePath, fileBuffer)

    // 2. Extract 1 frame per second using FFmpeg
    const framePaths = await extractFrames({
      videoPath: videoFilePath,
      outputDir: framesOutputDir,
      fps: 1,
    })

    if (framePaths.length === 0) {
      throw new Error('FFmpeg was unable to extract frames from the uploaded video.')
    }

    // Sort frames sequentially (frame_0001.jpg, frame_0002.jpg, ...)
    framePaths.sort()

    // 3. Prepare images for Gemini Multimodal API (sample up to 60 frames evenly if long)
    const maxFrames = 60
    const step = Math.max(1, Math.floor(framePaths.length / maxFrames))
    const sampledPaths = framePaths.filter((_, idx) => idx % step === 0)

    const imageParts = await Promise.all(
      sampledPaths.map(async (fPath) => {
        const frameBuffer = await fs.readFile(fPath)
        return {
          inlineData: {
            data: frameBuffer.toString('base64'),
            mimeType: 'image/jpeg',
          },
        }
      })
    )

    // 4. Define structured output schema
    const auditSchema: Schema = {
      type: 'OBJECT',
      properties: {
        matches: {
          type: 'ARRAY',
          items: {
            type: 'OBJECT',
            properties: {
              start_seconds: { type: 'NUMBER' },
              end_seconds: { type: 'NUMBER' },
              category: {
                type: 'STRING',
                description: 'PERSON, VEHICLE, OBJECT, SECURITY, or ANOMALY',
              },
              description: {
                type: 'STRING',
                description: 'Accurate description of what is occurring in the frame interval.',
              },
              confidence: {
                type: 'NUMBER',
                description: 'Confidence between 0.0 and 1.0 based strictly on visual clarity.',
              },
            },
            required: ['start_seconds', 'end_seconds', 'category', 'description', 'confidence'],
          },
        },
      },
      required: ['matches'],
    }

    // 5. Run Vision Model Inference
    const prompt = `You are an automated CCTV Video Surveillance Audit Engine.
The provided sequential images are sampled at exactly 1 frame every ${step} second(s) from a ${duration || framePaths.length}-second CCTV recording.
- Image index 0 corresponds to approximately 0 seconds.
- Image index i corresponds to timestamp (i * ${step}) seconds.

Target Audit Query: "${query}"

Strict Instructions:
1. Examine each frame carefully for the presence of the query target.
2. If the target is NOT visible in a time window, do NOT create an entry for it.
3. Identify continuous time windows (start_seconds to end_seconds) where the target appears.
4. Output valid JSON adhering strictly to the response schema.`

    const geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [...imageParts, { text: prompt }] }],
          generationConfig: {
            responseMimeType: 'application/json',
            responseSchema: auditSchema,
          },
        }),
      }
    )

    if (!geminiResponse.ok) {
      throw new Error(`Gemini API request failed (${geminiResponse.status}): ${await geminiResponse.text()}`)
    }

    const response = (await geminiResponse.json()) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>
    }
    const responseText = response.candidates?.[0]?.content?.parts
      ?.map((part) => part.text || '')
      .join('')

    const parsed = JSON.parse(responseText || '{"matches":[]}')
    const rawMatches = parsed.matches || []

    const formattedMatches: AuditMatch[] = rawMatches.map((m: RawMatch, index: number) => ({
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
      video_duration: duration || framePaths.length,
      query,
    }

    return NextResponse.json(responsePayload)
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error during analysis'
    console.error('API /api/analyze error:', err)
    return NextResponse.json({ error: message }, { status: 500 })
  } finally {
    // 6. Clean up temporary files
    if (tempDir) {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => { })
    }
  }
}