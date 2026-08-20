import { NextRequest, NextResponse } from 'next/server'
import { AuditMatch, AuditResponse } from '@/types/audit'
import { extractFrames } from '@/lib/ffmpeg'
import path from 'path'
import fs from 'fs/promises'
import os from 'os'

export const dynamic = 'force-dynamic'
export const maxDuration = 300

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
    const videoFile = formData.get('video') as File | null
    const query = (formData.get('query') as string) || ''
    const duration = parseFloat((formData.get('duration') as string) || '0')

    if (!videoFile) {
      return NextResponse.json({ error: 'No video file uploaded' }, { status: 400 })
    }

    // 1. Create temporary directory on host OS
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'cctv-audit-'))
    const videoFilePath = path.join(tempDir, videoFile.name || 'input.mp4')
    const framesOutputDir = path.join(tempDir, 'frames')

    // 2. Write file buffer to disk for FFmpeg
    const fileBuffer = Buffer.from(await videoFile.arrayBuffer())
    await fs.writeFile(videoFilePath, fileBuffer)

    // 3. Extract 1 frame per second using FFmpeg
    const framePaths = await extractFrames({
      videoPath: videoFilePath,
      outputDir: framesOutputDir,
      fps: 1,
    })

    if (framePaths.length === 0) {
      throw new Error('FFmpeg failed to extract frames from video.')
    }

    // 4. Try forwarding real extracted frame metadata or video to FastAPI backend
    try {
      const pyFormData = new FormData()
      pyFormData.append('query', query)
      pyFormData.append('duration', duration.toString())
      pyFormData.append('video', videoFile)

      const pyRes = await fetch('http://127.0.0.1:8000/api/analyze', {
        method: 'POST',
        body: pyFormData,
        signal: AbortSignal.timeout(300000),
      })

      if (pyRes.ok) {
        const data = await pyRes.json()
        return NextResponse.json(data)
      }
    } catch (backendError) {
      console.warn('FastAPI backend offline, processing locally or returning clear status:', backendError)
    }

    // 5. If FastAPI is offline, return a clear error or hook up a direct VLM API call here
    return NextResponse.json(
      {
        error: 'Analysis backend is unreachable. Start the FastAPI server on port 8000.',
        extractedFramesCount: framePaths.length,
      },
      { status: 503 }
    )
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error during extraction'
    return NextResponse.json({ error: message }, { status: 500 })
  } finally {
    // 6. Clean up temporary files
    if (tempDir) {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => { })
    }
  }
}