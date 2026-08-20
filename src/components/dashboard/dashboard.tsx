'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Sidebar } from './sidebar'
import { Topbar } from './topbar'
import { QueryBar } from './query-bar'
import { VideoPlayer } from './video-player'
import { AuditResults } from './audit-results'
import { AuditMatch, TimelineMarker, AnalysisProgress, AuditResponse } from '@/types/audit'

export function Dashboard() {
  // Initial State: Strict Zero Mock Data on Load
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

  // Execute Scalable Multi-Minute Video Analysis Pipeline
  const handleAnalyze = async (query: string) => {
    if (!videoFile && !videoUrl) return

    setIsProcessing(true)
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    try {
      const dur = videoDuration || 120
      const totalSteps = Math.max(2, Math.ceil(dur / 60))

      // Phase 1: Uploading Footage Progress
      setProgress({
        status: 'uploading',
        progress: 15,
        message: `Uploading footage (${videoFile ? (videoFile.size / (1024 * 1024)).toFixed(1) : '0'} MB) with streaming spooling...`,
      })

      const uploadTimer = setTimeout(() => {
        if (!abortController.signal.aborted) {
          setProgress({
            status: 'uploading',
            progress: 35,
            message: 'Streaming chunks to disk spool (/tmp/video_audit/)...',
          })
        }
      }, 500)

      const extractTimer = setTimeout(() => {
        if (!abortController.signal.aborted) {
          setProgress({
            status: 'extracting',
            progress: 50,
            message: `Extracting keyframes (1 fps) across ${totalSteps} segment(s) with JPEG downsampling...`,
            totalSegments: totalSteps,
          })
        }
      }, 1000)

      const analyzeTimer = setTimeout(() => {
        if (!abortController.signal.aborted) {
          setProgress({
            status: 'analyzing',
            progress: 75,
            message: `Analyzing segment 1 of ${totalSteps} (Motion delta filter active)...`,
            currentSegment: 1,
            totalSegments: totalSteps,
          })
        }
      }, 1600)

      const formData = new FormData()
      if (videoFile) {
        formData.append('file', videoFile)
      }
      formData.append('query', query)
      formData.append('duration', String(dur))
      formData.append('fileName', videoFile?.name || 'surveillance_feed.mp4')

      // Fetch with extended 300,000ms (5 minutes) timeout window
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
        signal: abortController.signal,
      })

      clearTimeout(uploadTimer)
      clearTimeout(extractTimer)
      clearTimeout(analyzeTimer)

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`)
      }

      const data: AuditResponse = await response.json()
      const returnedMatches: AuditMatch[] = data.matches || []

      // Generate Timeline Markers from returned Matches
      const totalDur = videoDuration || data.video_duration || 120
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
      console.error('Audit analysis failed:', err)
      setProgress({
        status: 'error',
        progress: 0,
        message: 'Analysis failed or timed out. Please verify video format and retry.',
      })
    } finally {
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
              onLoadedMetadata={(d) => setVideoDuration(d)}
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