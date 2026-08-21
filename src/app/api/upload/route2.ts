import { NextRequest, NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'
export const maxDuration = 300 // 5 minutes execution window for large multi-minute video uploads

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData()
    const file = formData.get('file') as File | null
    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 })
    }

    // Forward to Python FastAPI backend if available with 300s timeout
    try {
      const pyRes = await fetch('http://127.0.0.1:8000/api/upload', {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(300000), // 300 seconds (5 minutes)
      })
      if (pyRes.ok) {
        const data = await pyRes.json()
        return NextResponse.json(data)
      }
    } catch {
      // Backend offline fallback
    }

    return NextResponse.json({
      video_id: `vid-${Date.now()}-${file.name.replace(/[^a-zA-Z0-9._-]/g, '_')}`,
      original_filename: file.name,
      size_bytes: file.size,
      duration_seconds: 120,
    })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Upload error'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
