'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Sidebar } from './sidebar'
import { Topbar } from './topbar'
import { QueryBar } from './query-bar'
import { VideoPlayer } from './video-player'
import { AuditResults } from './audit-results'
import { AuditMatch, TimelineMarker, AnalysisProgress, AuditResponse } from '@/types/audit'
import { preloadFFmpeg } from '@/lib/ffmpeg-preload'
import { compressVideo } from '@/lib/compress'

export function Dashboard() {
  // Initial State: Strict Zero Mock Data on Load
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [videoDuration, setVideoDuration] = useState<number>(0)
  const [videoMeta, setVideoMeta] = useState<{ width: number; height: number; fps: number } | null>(null)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [matches, setMatches] = useState<AuditMatch[]>([])
  const [markers, setMarkers] = useState<TimelineMarker[]>([])
  const [activeMatchId, setActiveMatchId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [progress, setProgress] = useState<AnalysisProgress | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Preload FFmpeg.wasm in the background when dashboard mounts
  useEffect(() => {
    preloadFFmpeg()
  }, [])

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

  // Clean up Object URL on unmount or file change
  useEffect(() => {
    return () => {
      if (videoUrl && videoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(videoUrl)
      }
    }
  }, [videoUrl])

  // Handle Video File Upload
  const handleFileUpload = useCallback((file: File) => {
    // Check 500MB maximum body limit
    const MAX_SIZE_BYTES = 500 * 1024 * 1024
    if (file.size > MAX_SIZE_BYTES) {
      alert(`File size exceeds the 500MB maximum limit (${(file.size / (1024 * 1024)).toFixed(1)}MB). Please select a compressed file.`)
      return
    }

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
    setVideoMeta(null) // Reset metadata on new file
  }, [videoUrl])

  // Handle Seeking & Playhead Synchronization
  const handleSeek = useCallback((seconds: number) => {
    setCurrentTime(seconds)
  }, [])

  // Handle Timeline Marker Click - Seek & Resume Active Continuous Playback
  const handleSelectMarker = useCallback((id: string, seconds: number) => {
    setActiveMatchId(id)
    setCurrentTime(seconds)
    setIsPlaying(true)
  }, [])

  // Handle Audit Result Card Click - Seek & Resume Active Continuous Playback
  const handleSelectMatch = useCallback((id: string, startSeconds: number) => {
    setActiveMatchId(id)
    setCurrentTime(startSeconds)
    setIsPlaying(true)
  }, [])

  // Handle Video Play / Pause Toggle
  const handlePlayPause = useCallback(() => {
    setIsPlaying((prev) => !prev)
  }, [])

  // Cancel Analysis
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

  // Execute Video Analysis Pipeline
  const handleAnalyze = async (query: string) => {
    if (!videoFile && !videoUrl) {
      alert('Please upload a video file first.')
      return
    }

    setIsProcessing(true)
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    let uploadTimer: NodeJS.Timeout | null = null
    let extractTimer: NodeJS.Timeout | null = null

    try {
      const dur = videoDuration || 120

      // Ensure a valid binary File exists
      let fileToSend = videoFile
      if (!fileToSend && videoUrl) {
        const blobRes = await fetch(videoUrl)
        const blob = await blobRes.blob()
        fileToSend = new File([blob], 'footage.mp4', { type: blob.type || 'video/mp4' })
      }

      if (!fileToSend) {
        throw new Error('No valid video file could be prepared for analysis.')
      }

      // Step 0: Client-side compression (non-blocking Web Worker)
      let uploadFile = fileToSend
      if (videoMeta) {
        try {
          setProgress({
            status: 'compressing',
            progress: 0,
            message: 'Optimizing video for upload...',
          })

          const compressed = await compressVideo({
            file: fileToSend,
            duration: dur,
            width: videoMeta.width,
            height: videoMeta.height,
            fps: videoMeta.fps,
            onProgress: (pct) => {
              if (!abortController.signal.aborted) {
                setProgress({
                  status: 'compressing',
                  progress: pct,
                  message: `Compressing video... ${pct}%`,
                })
              }
            },
          })

          if (!abortController.signal.aborted) {
            uploadFile = new File([compressed.blob], compressed.filename, {
              type: 'video/mp4',
            })
            console.log(
              `[Sentinel] Compressed: ${compressed.reduction}% reduction ` +
              `(${(compressed.originalSize / 1e6).toFixed(1)}MB → ${(compressed.compressedSize / 1e6).toFixed(1)}MB)`,
            )
          }
        } catch (compressErr) {
          console.warn('[Sentinel] Compression failed, uploading original:', compressErr)
          // Fall back to original file
        }
      }

      // Step 1: Upload progress
      if (!abortController.signal.aborted) {
        setProgress({
          status: 'uploading',
          progress: 20,
          message: `Uploading footage (${(uploadFile.size / (1024 * 1024)).toFixed(1)} MB)...`,
        })
      }

      uploadTimer = setTimeout(() => {
        if (!abortController.signal.aborted) {
          setProgress({
            status: 'extracting',
            progress: 50,
            message: 'Extracting video frames via FFmpeg...',
          })
        }
      }, 600)

      extractTimer = setTimeout(() => {
        if (!abortController.signal.aborted) {
          setProgress({
            status: 'analyzing',
            progress: 75,
            message: 'Multimodal AI analyzing visual patterns...',
          })
        }
      }, 1500)

      // Build payload containing expected keys
      const formData = new FormData()
      formData.append('video', uploadFile)
      formData.append('file', uploadFile) // Fallback alias
      formData.append('query', query)
      formData.append('duration', dur.toString())
      formData.append('fileName', uploadFile.name)

      // Send to Next.js API Route
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
        signal: abortController.signal,
      })

      if (uploadTimer) clearTimeout(uploadTimer)
      if (extractTimer) clearTimeout(extractTimer)

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}))
        throw new Error(errJson.error || `Server returned ${response.status}: ${response.statusText}`)
      }

      const data: AuditResponse = await response.json()
      const returnedMatches: AuditMatch[] = data.matches || []

      // Generate visual timeline markers
      const totalDur = videoDuration || data.video_duration || dur
      const generatedMarkers: TimelineMarker[] = returnedMatches.map((m) => {
        const cat = (m.category || 'ANOMALY').toUpperCase()
        let color: 'primary' | 'accent' | 'destructive' = 'primary'
        if (cat === 'VEHICLE') color = 'accent'
        if (cat === 'ANOMALY' || cat === 'SECURITY') color = 'destructive'

        return {
          id: m.id,
          position: totalDur > 0 ? (m.start_seconds / totalDur) * 100 : 0,
          seconds: m.start_seconds,
          label: m.description,
          color,
          category: cat,
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
        message: `Audit complete. Identified ${returnedMatches.length} matching surveillance event(s).`,
      })
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }
      const message = err instanceof Error ? err.message : 'Analysis failed. Please verify video format and retry.'
      console.error('Audit analysis failed:', err)
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
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <Sidebar />
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        <Topbar />

        <main className="flex-1 overflow-y-auto p-4 lg:p-6 space-y-4">
          <QueryBar
            onAnalyze={handleAnalyze}
            onFileUpload={handleFileUpload}
            onCancel={handleCancelAnalysis}
            isProcessing={isProcessing}
            hasVideo={Boolean(videoUrl)}
            currentFileName={videoFile?.name}
          />

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
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
              onLoadedMetadata={(d, meta) => {
                setVideoDuration(d)
                if (meta) setVideoMeta(meta)
              }}
            />

            <div className="h-[450px] lg:h-[calc(100vh-14rem)]">
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
    </div>
  )
}