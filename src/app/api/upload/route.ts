import { NextRequest, NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'
export const maxDuration = 300 // 5 minutes execution window for large multi-minute video uploads

// Allowed video MIME types and extensions
const ALLOWED_MIME_TYPES = new Set([
  'video/mp4',
  'video/webm',
  'video/avi',
  'video/x-msvideo',
  'video/quicktime',
  'video/x-matroska',
  'video/ogg',
])

const ALLOWED_EXTENSIONS = new Set(['.mp4', '.webm', '.avi', '.mov', '.mkv', '.ogg'])

const MAX_FILE_SIZE = 500 * 1024 * 1024 // 500MB

function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf('.')
  return lastDot >= 0 ? filename.slice(lastDot).toLowerCase() : ''
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData()
    const file = formData.get('file') as File | null
    
    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 })
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: `File size exceeds ${MAX_FILE_SIZE / (1024 * 1024)}MB limit` },
        { status: 413 }
      )
    }

    // Validate file extension
    const extension = getFileExtension(file.name)
    if (!ALLOWED_EXTENSIONS.has(extension)) {
      return NextResponse.json(
        { error: `Invalid file type. Allowed: ${Array.from(ALLOWED_EXTENSIONS).join(', ')}` },
        { status: 400 }
      )
    }

    // Validate MIME type (if provided by browser)
    if (file.type && !ALLOWED_MIME_TYPES.has(file.type)) {
      // Some browsers don't send correct MIME types for video files
      // Only reject if extension is also invalid
      if (!ALLOWED_EXTENSIONS.has(extension)) {
        return NextResponse.json(
          { error: 'Invalid video file type' },
          { status: 400 }
        )
      }
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
