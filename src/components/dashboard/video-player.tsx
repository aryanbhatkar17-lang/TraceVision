'use client'

import { useRef, useEffect, useState, DragEvent } from 'react'
import { Play, Pause, Volume2, VolumeX, Maximize2, UploadCloud, VideoOff, Radio } from 'lucide-react'
import { cn } from '@/lib/utils'
import { TimelineMarker } from '@/types/audit'

interface VideoPlayerProps {
  videoUrl: string | null
  videoName?: string
  duration: number
  currentTime: number
  isPlaying: boolean
  markers: TimelineMarker[]
  activeMarkerId: string | null
  onPlayPause: () => void
  onSeek: (seconds: number) => void
  onSelectMarker: (id: string, seconds: number) => void
  onFileUpload: (file: File) => void
  onTimeUpdate?: (seconds: number) => void
  onLoadedMetadata?: (duration: number, meta?: { width: number; height: number; fps: number }) => void
}

function formatTime(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return '00:00'
  const hrs = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  if (hrs > 0) {
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

export function VideoPlayer({
  videoUrl,
  videoName,
  duration,
  currentTime,
  isPlaying,
  markers,
  activeMarkerId,
  onPlayPause,
  onSeek,
  onSelectMarker,
  onFileUpload,
  onTimeUpdate,
  onLoadedMetadata,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [isMuted, setIsMuted] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  // Synchronize playback and seeking with native HTML5 video
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    if (isPlaying) {
      const playPromise = video.play()
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          console.warn('Playback resume handled:', err)
        })
      }
    } else {
      video.pause()
    }
  }, [isPlaying])

  // Synchronize external timestamp seeking and maintain smooth continuous playback
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    if (Math.abs(video.currentTime - currentTime) > 0.4) {
      video.currentTime = currentTime
      if (isPlaying) {
        video.play().catch(() => {})
      }
    }
  }, [currentTime, isPlaying])

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('video/') || /\.(mp4|avi|webm|mov|mkv)$/i.test(file.name)) {
        onFileUpload(file)
      }
    }
  }

  const handleToggleFullscreen = () => {
    if (!videoRef.current) return
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {})
    } else {
      videoRef.current.requestFullscreen().catch(() => {})
    }
  }

  const progressPercent = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0

  return (
    <div className="flex flex-col gap-3">
      {/* Screen Viewport Frame */}
      <div className="relative flex flex-col rounded-lg border border-border bg-[#05070a] overflow-hidden shadow-2xl">
        {/* Top HUD Telemetry Overlay (pointer-events: none so it never intercepts user clicks) */}
        <div className="pointer-events-none absolute top-3 left-3 z-20 flex items-center gap-2">
          {videoUrl ? (
            <span className="flex items-center gap-2 rounded bg-black/85 px-2.5 py-1 font-mono text-[11px] text-red-400 border border-red-500/40 backdrop-blur-md">
              <span className={cn("size-1.5 rounded-full bg-red-500", isPlaying ? "animate-pulse" : "opacity-80")} />
              {`${isPlaying ? 'LIVE STREAM' : 'PAUSED'} // ${videoName ? videoName.slice(0, 18) : 'CAM-01'}`}
            </span>
          ) : (
            <span className="flex items-center gap-2 rounded bg-black/85 px-2.5 py-1 font-mono text-[11px] text-muted-foreground border border-border backdrop-blur-md">
              <Radio className="size-3 text-muted-foreground" />
              {'STANDBY // NO FEED'}
            </span>
          )}
        </div>

        <div className="pointer-events-none absolute top-3 right-3 z-20 flex items-center gap-2">
          {videoUrl ? (
            <span className="flex items-center gap-1.5 rounded bg-black/85 px-2.5 py-1 font-mono text-[11px] text-primary border border-primary/40 backdrop-blur-md">
              AI_TRACKING: {isPlaying ? 'STREAMING' : 'READY'}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 rounded bg-black/85 px-2.5 py-1 font-mono text-[11px] text-muted-foreground border border-border backdrop-blur-md">
              INPUT: DISCONNECTED
            </span>
          )}
        </div>

        {/* Video Viewport / Empty State Dropzone */}
        <div className="relative aspect-video w-full flex items-center justify-center bg-[#020305] overflow-hidden">
          {/* Engineering grid blueprint overlay */}
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#1a223015_1px,transparent_1px),linear-gradient(to_bottom,#1a223015_1px,transparent_1px)] bg-[size:2.5rem_2.5rem]" />

          {videoUrl ? (
            <>
              <video
                ref={videoRef}
                src={videoUrl}
                playsInline
                controls={false}
                muted={isMuted}
                onTimeUpdate={(e) => {
                  const curr = e.currentTarget.currentTime
                  if (onTimeUpdate) onTimeUpdate(curr)
                }}
                onLoadedMetadata={(e) => {
                  const video = e.currentTarget
                  const dur = video.duration
                  if (onLoadedMetadata && !isNaN(dur)) {
                    // Extract video metadata for compression profiling
                    const meta = {
                      width: video.videoWidth || 0,
                      height: video.videoHeight || 0,
                      fps: 30, // Default — browsers don't expose native fps
                    }
                    onLoadedMetadata(dur, meta)
                  }
                }}
                onEnded={() => {
                  if (isPlaying) onPlayPause()
                }}
                className="h-full w-full object-contain cursor-pointer"
                onClick={onPlayPause}
              />

              {/* Central Play Overlay Trigger on pause */}
              {!isPlaying && (
                <button
                  type="button"
                  onClick={onPlayPause}
                  className="group absolute z-10 flex size-14 items-center justify-center rounded-full bg-primary/20 border border-primary/60 text-primary transition-all hover:scale-110 hover:bg-primary/30 active:scale-95 shadow-[0_0_25px_rgba(16,185,129,0.4)] cursor-pointer"
                  aria-label="Play Video"
                >
                  <Play className="size-6 translate-x-0.5 fill-primary/30" />
                </button>
              )}

              {/* Bottom Video HUD Overlay (pointer-events: none, non-blocking telemetry) */}
              <div className="pointer-events-none absolute bottom-3 left-3 z-10 font-mono text-[10px] text-slate-400/80 tracking-widest bg-black/70 px-2 py-0.5 rounded border border-border/50">
                {`${isPlaying ? 'ACTIVE_PLAYBACK // 1.0X' : 'SEEK_SYNC: READY'} // DURATION: ${formatTime(duration)}`}
              </div>
            </>
          ) : (
            /* Tactical Empty-State Dropzone */
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                'group relative flex h-full w-full cursor-pointer flex-col items-center justify-center p-6 text-center transition-all duration-200 border-2 border-dashed m-2 rounded-md',
                isDragging
                  ? 'border-primary bg-primary/10 scale-[0.99]'
                  : 'border-border/80 hover:border-primary/60 hover:bg-card/30'
              )}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/webm,video/avi,video/x-msvideo,video/quicktime,video/x-matroska"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) onFileUpload(file)
                }}
              />

              <div className="flex size-14 items-center justify-center rounded-2xl bg-secondary/80 border border-border/80 text-muted-foreground transition-all duration-300 group-hover:border-primary/50 group-hover:text-primary group-hover:scale-105 group-hover:shadow-[0_0_20px_rgba(16,185,129,0.2)]">
                {isDragging ? (
                  <UploadCloud className="size-7 text-primary animate-bounce" />
                ) : (
                  <VideoOff className="size-7" />
                )}
              </div>

              <div className="mt-4 max-w-sm space-y-1.5">
                <h3 className="font-semibold text-sm text-foreground tracking-wide">
                  No CCTV Feed Active
                </h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Drag and drop or click <span className="text-primary font-medium underline underline-offset-2">Upload Footage</span> to begin audit.
                </p>
              </div>

              <div className="mt-3 flex items-center gap-2 text-[10px] font-mono text-muted-foreground/70 uppercase tracking-wider">
                <span>Supports MP4</span>
                <span>•</span>
                <span>WEBM</span>
                <span>•</span>
                <span>AVI</span>
                <span>•</span>
                <span>MKV</span>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Hardware Control Bar */}
        <div className="flex h-11 items-center justify-between px-3 border-t border-border bg-card">
          <div className="flex items-center gap-3">
            <button
              onClick={onPlayPause}
              disabled={!videoUrl}
              className="text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              aria-label={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
            </button>
            <span className="font-mono text-xs text-muted-foreground">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center gap-3 text-muted-foreground">
            <button
              onClick={() => setIsMuted(!isMuted)}
              disabled={!videoUrl}
              className="hover:text-foreground transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              aria-label={isMuted ? 'Unmute' : 'Mute'}
            >
              {isMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
            </button>
            <button
              onClick={handleToggleFullscreen}
              disabled={!videoUrl}
              className="hover:text-foreground transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              aria-label="Fullscreen"
            >
              <Maximize2 className="size-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Timeline Controls */}
      <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3 shadow-lg">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] tracking-widest text-muted-foreground uppercase">
            Timeline Index
          </span>
          <span className="font-mono text-[10px] text-muted-foreground">
            {`${markers.length} ${markers.length === 1 ? 'EVENT FLAGGED' : 'EVENTS FLAGGED'}`}
          </span>
        </div>

        <div className="relative py-2.5 select-none">
          {/* Progress Track */}
          <div className="relative h-2 w-full rounded-full bg-secondary overflow-hidden border border-border/40">
            <div
              className="absolute top-0 left-0 h-full bg-primary/60 transition-[width] duration-75"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          {/* Interactive Range Input */}
          <input
            type="range"
            min={0}
            max={duration > 0 ? duration : 100}
            step={0.1}
            value={currentTime}
            disabled={!videoUrl}
            onChange={(e) => {
              const val = Number(e.target.value)
              onSeek(val)
            }}
            className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-20 disabled:cursor-not-allowed"
            aria-label="Video timeline scrub"
          />

          {/* Playhead Indicator */}
          {videoUrl && (
            <div
              className="absolute top-1/2 -translate-y-1/2 size-3.5 -ml-[7px] rounded-full bg-foreground border-2 border-primary pointer-events-none z-10 shadow-[0_0_8px_var(--primary)]"
              style={{ left: `${progressPercent}%` }}
            />
          )}

          {/* Flagged Markers */}
          {markers.map((marker) => {
            const isActive = activeMarkerId === marker.id
            const pos = duration > 0 ? (marker.seconds / duration) * 100 : marker.position
            return (
              <button
                key={marker.id}
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onSelectMarker(marker.id, marker.seconds)
                }}
                style={{ left: `${Math.max(1, Math.min(99, pos))}%` }}
                className={cn(
                  'absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border transition-all hover:scale-150 z-10 cursor-pointer',
                  marker.color === 'primary'
                    ? 'bg-emerald-500 border-emerald-300 shadow-[0_0_8px_rgba(16,185,129,0.8)]'
                    : marker.color === 'destructive'
                    ? 'bg-red-500 border-red-300 shadow-[0_0_8px_rgba(239,68,68,0.8)]'
                    : 'bg-amber-500 border-amber-300 shadow-[0_0_8px_rgba(245,158,11,0.8)]',
                  isActive
                    ? 'ring-2 ring-primary ring-offset-2 ring-offset-background scale-150 z-20'
                    : 'opacity-90'
                )}
                title={`${marker.label} (${formatTime(marker.seconds)})`}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}