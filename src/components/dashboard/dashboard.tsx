'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { QueryBar } from './query-bar'
import { VideoPlayer } from './video-player'
import { AuditResults } from './audit-results'
import { AuditMatch, TimelineMarker, AnalysisProgress, AuditResponse } from '@/types/audit'
import { ShieldCheck, RotateCcw } from 'lucide-react'

// Direct-to-Render backend URL.
// The browser uploads straight to Render — bypassing Vercel entirely and
// avoiding the 4.5 MB Serverless Function body limit. CORS is open on
// the backend with allow_origins=["*"] so this is safe.
const API_BASE_URL = "https://sih-2026-6ifa.onrender.com"

export default function Dashboard() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [videoDuration, setVideoDuration] = useState<number>(0)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)

  const [matches, setMatches] = useState<AuditMatch[]>([])
  const [markers, setMarkers] = useState<TimelineMarker[]>([])
  const [activeMatchId, setActiveMatchId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [progress, setProgress] = useState<AnalysisProgress | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)


  // Clear any residual session / local state on initial mount
  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        window.localStorage?.removeItem('sentinel_audit_state')
        window.sessionStorage?.removeItem('sentinel_audit_state')
      }
    } catch {
      // Ignore storage access errors
    }
  }, [])

  useEffect(() => {
    return () => {
      if (videoUrl && videoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(videoUrl)
      }
    }
  }, [videoUrl])

  const handleFileUpload = useCallback((file: File) => {
    if (videoUrl && videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(videoUrl)
    }
    const url = URL.createObjectURL(file)
    setVideoFile(file)
    setVideoUrl(url)
    setCurrentTime(0)
    setIsPlaying(false)
    setMatches([])
    setMarkers([])
    setActiveMatchId(null)
    setProgress(null)
  }, [videoUrl])

  const handleReset = useCallback(() => {
    if (videoUrl && videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(videoUrl)
    }
    setVideoFile(null)
    setVideoUrl(null)
    setVideoDuration(0)
    setCurrentTime(0)
    setIsPlaying(false)
    setMatches([])
    setMarkers([])
    setActiveMatchId(null)
    setProgress(null)
  }, [videoUrl])

  const handleSeek = useCallback((seconds: number) => {
    setCurrentTime(seconds)
  }, [])

  const handleSelectMarker = useCallback((id: string, seconds: number) => {
    setActiveMatchId(id)
    setCurrentTime(seconds)
    setIsPlaying(true)
  }, [])

  const handleSelectMatch = useCallback((id: string, startSeconds: number) => {
    setActiveMatchId(id)
    setCurrentTime(startSeconds)
    setIsPlaying(true)
  }, [])

  const handlePlayPause = useCallback(() => {
    setIsPlaying((prev) => !prev)
  }, [])

  const handleCancelAnalysis = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsProcessing(false)
    setProgress({
      status: 'cancelled',
      progress: 0,
      message: 'Analysis cancelled by operator.',
    })
  }, [])

  const handleAnalyze = async (query: string) => {
    if (!videoFile && !videoUrl) {
      alert('Please load a video recording first.')
      return
    }

    // Reset markers immediately when a new query starts
    setMatches([])
    setMarkers([])
    setActiveMatchId(null)

    setIsProcessing(true)
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    let uploadTimer: ReturnType<typeof setTimeout> | null = null
    let extractTimer: ReturnType<typeof setTimeout> | null = null

    try {
      const dur = videoDuration || 120

      // Resolve a File object from videoFile or videoUrl
      let fileToSend = videoFile
      if (!fileToSend && videoUrl) {
        const blobRes = await fetch(videoUrl)
        const blob = await blobRes.blob()
        fileToSend = new File([blob], 'footage.mp4', { type: blob.type || 'video/mp4' })
      }

      if (!fileToSend) {
        throw new Error('No valid video file could be prepared for analysis.')
      }

      // Stream the raw file directly — no client-side compression.
      // The backend runs FFmpeg with CUDA NVDEC hardware acceleration
      // to extract high-fidelity 1024px frames at the server level.
      if (!abortController.signal.aborted) {
        setProgress({
          status: 'uploading',
          progress: 15,
          message: `Uploading footage (${(fileToSend.size / (1024 * 1024)).toFixed(1)} MB)...`,
        })
      }

      uploadTimer = setTimeout(() => {
        if (!abortController.signal.aborted) {
          setProgress({
            status: 'extracting',
            progress: 45,
            message: 'Server extracting frames with GPU acceleration...',
          })
        }
      }, 800)

      extractTimer = setTimeout(() => {
        if (!abortController.signal.aborted) {
          setProgress({
            status: 'analyzing',
            progress: 75,
            message: 'Multimodal AI analyzing spatial patterns...',
          })
        }
      }, 2500)

      const formData = new FormData()
      formData.append('file', fileToSend)
      formData.append('query', query)
      formData.append('duration', dur.toString())
      formData.append('fileName', fileToSend.name)

      // Use the modern api.py endpoint, not the legacy heuristic mock endpoint!
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
        signal: abortController.signal,
      })

      if (uploadTimer) clearTimeout(uploadTimer)
      if (extractTimer) clearTimeout(extractTimer)

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}))
        throw new Error(errJson.detail || errJson.error || `Server returned ${response.status}: ${response.statusText}`)
      }

      const data: AuditResponse = await response.json()
      const returnedMatches: AuditMatch[] = data.matches || []

      const totalDur = videoDuration || data.video_duration || dur
      const generatedMarkers: TimelineMarker[] = returnedMatches.map((m) => {
        return {
          id: m.id,
          position: totalDur > 0 ? (m.start_seconds / totalDur) * 100 : 0,
          seconds: m.start_seconds,
          label: m.description,
          color: 'primary', // Always green colored timeline markers
          category: (m.category || 'ANOMALY').toUpperCase(),
        }
      })

      setMatches(returnedMatches)
      setMarkers(generatedMarkers)
      if (returnedMatches.length > 0) {
        setActiveMatchId(returnedMatches[0].id)
      }

      setProgress({
        status: 'completed',
        progress: 100,
        message: `Identified ${returnedMatches.length} matching surveillance event(s).`,
      })
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }
      // Force empty state on failure (destroying any fallback/dummy data traces)
      setMatches([])
      setMarkers([])
      
      const message = err instanceof Error ? err.message : 'Analysis failed. Please check video format and retry.'
      console.error('Analysis error:', err)
      setProgress({
        status: 'error',
        progress: 0,
        message,
      })
      alert(message)
    } finally {
      if (uploadTimer) clearTimeout(uploadTimer)
      if (extractTimer) clearTimeout(extractTimer)
      setIsProcessing(false)
      abortControllerRef.current = null
    }
  }

  return (
    <div className="min-h-screen bg-[#f0f4f8] text-slate-900 flex flex-col font-sans selection:bg-blue-100 selection:text-blue-900">
      {/* 1. Integrated Header */}
      <header className="sticky top-0 z-50 bg-white/70 backdrop-blur-xl border-b border-slate-200/90 shadow-[0_4px_20px_rgba(0,0,0,0.03)] px-6 lg:px-10 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-3 group">
          <div className="relative w-7 h-7 flex items-center justify-center rounded-lg overflow-hidden shrink-0 shadow-sm bg-slate-900">
            <Image
              src="/tracevision-icon.svg"
              alt="TraceVision Icon"
              width={28}
              height={28}
              className="object-contain"
              priority
            />
          </div>
          <span className="text-sm font-bold tracking-wider uppercase text-slate-800 group-hover:text-blue-700 transition-colors">
            TraceVision
          </span>
        </Link>

        <div className="flex items-center gap-3">
          {videoUrl && (
            <button
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold text-slate-600 hover:text-slate-900 bg-white border border-slate-200 rounded-lg shadow-2xs hover:bg-slate-50 transition cursor-pointer"
              title="Reset Video and Search"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Feed
            </button>
          )}

          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium text-slate-700 bg-white/60 border border-slate-200 shadow-2xs">
            <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
            Console Active
          </span>
        </div>
      </header>

      {/* 2. Main Workbench Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-4">
        <QueryBar
          onAnalyze={handleAnalyze}
          onFileUpload={handleFileUpload}
          onCancel={handleCancelAnalysis}
          isProcessing={isProcessing}
          hasVideo={Boolean(videoUrl)}
          currentFileName={videoFile?.name}
        />

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_380px] items-start">
          <div className="w-full">
            <VideoPlayer
              videoUrl={videoUrl}
              videoName={videoFile?.name}
              duration={videoDuration}
              currentTime={currentTime}
              isPlaying={isPlaying}
              markers={markers}
              activeMarkerId={activeMatchId}
              onPlayPause={handlePlayPause}
              onSeek={handleSeek}
              onSelectMarker={handleSelectMarker}
              onFileUpload={handleFileUpload}
              onTimeUpdate={(t) => setCurrentTime(t)}
              onLoadedMetadata={(d) => {
                setVideoDuration(d)
              }}
            />
          </div>

          <div className="w-full sticky top-20">
            <AuditResults
              matches={matches}
              isProcessing={isProcessing}
              progress={progress}
              activeMatchId={activeMatchId}
              onSelect={handleSelectMatch}
              onCancel={handleCancelAnalysis}
            />
          </div>
        </div>
      </main>
    </div>
  )
}