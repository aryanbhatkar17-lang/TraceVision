import { NextRequest, NextResponse } from 'next/server'
import { AuditMatch, AuditResponse } from '@/types/audit'

// Next.js Route Segment Config
export const dynamic = 'force-dynamic'
export const maxDuration = 300

// Python backend URL — all heavy processing (OpenCV, FFmpeg, CLIP, Smoothing) runs here
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

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
  try {
    // -----------------------------------------------------------------------
    // Step 1: Parse incoming form data from the dashboard
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Step 2: Upload the video to the Python backend
    // -----------------------------------------------------------------------
    const uploadForm = new FormData()
    uploadForm.append('file', videoFile, videoFile.name)

    const uploadRes = await fetch(`${BACKEND_URL}/api/upload`, {
      method: 'POST',
      body: uploadForm,
      signal: AbortSignal.timeout(120_000), // 2 min upload timeout
    })

    if (!uploadRes.ok) {
      const err = await uploadRes.text()
      throw new Error(`Upload to backend failed (${uploadRes.status}): ${err}`)
    }

    const uploadData = await uploadRes.json()
    const videoId: string = uploadData.video_id
    const videoDuration: number = uploadData.duration_seconds || duration

    // -----------------------------------------------------------------------
    // Step 3: Run analysis on the backend (handles OpenCV keyframes,
    //         motion filtering, CLIP/Semantic auditing, and Temporal Smoothing)
    // -----------------------------------------------------------------------
    const analyzeForm = new FormData()
    analyzeForm.append('video_id', videoId)
    analyzeForm.append('query', query)
    analyzeForm.append('duration', String(videoDuration))

    const analyzeRes = await fetch(`${BACKEND_URL}/api/analyze`, {
      method: 'POST',
      body: analyzeForm,
      signal: AbortSignal.timeout(240_000), // 4 min analysis timeout
    })

    if (!analyzeRes.ok) {
      const err = await analyzeRes.text()
      throw new Error(`Analysis backend failed (${analyzeRes.status}): ${err}`)
    }

    const analysisData = await analyzeRes.json()

    // -----------------------------------------------------------------------
    // Step 4: Map backend response to the AuditResponse schema
    // -----------------------------------------------------------------------
    const rawMatches: AuditMatch[] = (analysisData.matches || []).map(
      (m: any, index: number): AuditMatch => ({
        id: m.id || `match-${index + 1}`,
        start_time: m.start_time || formatSeconds(m.start_seconds),
        end_time: m.end_time || formatSeconds(m.end_seconds),
        start_seconds: Number(m.start_seconds) || 0,
        end_seconds: Number(m.end_seconds) || 0,
        category: m.category || 'PERSON',
        description: m.description,
        confidence: Number(m.confidence?.toFixed?.(2)) || 0.9,
        chunk_id: m.chunk_id,
      })
    )

    const response: AuditResponse = {
      matches: rawMatches,
      total_chunks: analysisData.total_chunks || 1,
      video_duration: analysisData.video_duration || videoDuration,
      query,
    }

    return NextResponse.json(response)
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error during analysis'
    console.error('[/api/analyze] Error:', err)
    return NextResponse.json({ error: message }, { status: 500 })
  }
}