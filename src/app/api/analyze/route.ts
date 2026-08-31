import { NextRequest, NextResponse } from 'next/server'
import { AuditMatch, AuditResponse } from '@/types/audit'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

// Interface for backend raw match objects
interface RawBackendMatch {
  id?: string
  start_time?: string
  end_time?: string
  start_seconds?: number | string
  end_seconds?: number | string
  category?: string
  description?: string
  confidence?: number
  chunk_id?: string | number
}

function formatSeconds(sec: number): string {
  const hrs = Math.floor(sec / 3600)
  const mins = Math.floor((sec % 3600) / 60)
  const secs = Math.floor(sec % 60)
  if (hrs > 0) {
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

async function fetchWithTimeout(url: string, options: RequestInit, timeoutMs = 45000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    return response
  } finally {
    clearTimeout(timer)
  }
}

export async function POST(req: NextRequest) {
  try {
    if (!process.env.BACKEND_URL && process.env.NODE_ENV === 'production') {
      return NextResponse.json(
        { error: 'BACKEND_URL environment variable is not configured on Vercel.' },
        { status: 500 }
      )
    }

    const formData = await req.formData()
    const rawFile = formData.get('video') || formData.get('file')
    const query = (formData.get('query') as string) || ''
    const duration = parseFloat((formData.get('duration') as string) || '0')

    if (!rawFile || typeof rawFile === 'string') {
      return NextResponse.json(
        { error: 'No video file provided for analysis.' },
        { status: 400 }
      )
    }

    const videoFile = rawFile as File

    // Step 1: Upload video to Render
    const uploadForm = new FormData()
    uploadForm.append('file', videoFile, videoFile.name)

    const uploadRes = await fetchWithTimeout(`${BACKEND_URL}/api/upload`, {
      method: 'POST',
      body: uploadForm,
    }).catch((err) => {
      throw new Error(`Failed to connect to Render backend at ${BACKEND_URL}. Server might be sleeping or down. (${err.message})`)
    })

    if (!uploadRes.ok) {
      const err = await uploadRes.text()
      throw new Error(`Upload to backend failed (${uploadRes.status}): ${err}`)
    }

    const uploadData = await uploadRes.json()
    const videoId: string = uploadData.video_id
    const videoDuration: number = uploadData.duration_seconds || duration

    // Step 2: Trigger Analysis
    const analyzeForm = new FormData()
    analyzeForm.append('video_id', videoId)
    analyzeForm.append('query', query)
    analyzeForm.append('duration', String(videoDuration))

    const analyzeRes = await fetchWithTimeout(`${BACKEND_URL}/api/analyze`, {
      method: 'POST',
      body: analyzeForm,
    }).catch((err) => {
      throw new Error(`Analysis request to backend timed out or failed. (${err.message})`)
    })

    if (!analyzeRes.ok) {
      const err = await analyzeRes.text()
      throw new Error(`Analysis backend failed (${analyzeRes.status}): ${err}`)
    }

    const analysisData = await analyzeRes.json()

    // Step 3: Map backend response with typed interface
    const rawMatches: AuditMatch[] = (analysisData.matches || []).map(
      (m: RawBackendMatch, index: number): AuditMatch => {
        const startSec = Number(m.start_seconds) || 0
        const endSec = Number(m.end_seconds) || 0

        return {
          id: m.id || `match-${index + 1}`,
          start_time: m.start_time || formatSeconds(startSec),
          end_time: m.end_time || formatSeconds(endSec),
          start_seconds: startSec,
          end_seconds: endSec,
          category: m.category || 'PERSON',
          description: m.description || '',
          confidence: typeof m.confidence === 'number' ? Number(m.confidence.toFixed(2)) : 0.9,
          chunk_id: m.chunk_id !== undefined ? String(m.chunk_id) : undefined,
        }
      }
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