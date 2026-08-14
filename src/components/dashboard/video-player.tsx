'use client'

import { useState } from 'react'
import { Play, Pause, Volume2, Maximize2, ShieldAlert } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Marker {
  id: string
  position: number
  label: string
  color: 'primary' | 'accent'
}

interface VideoPlayerProps {
  markers: Marker[]
  activeMarkerId: string
  onScrub: (id: string) => void
}

export function VideoPlayer({ markers, activeMarkerId, onScrub }: VideoPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(7403)
  const totalDuration = 9717

  const progressPercent = (currentTime / totalDuration) * 100

  return (
    <div className="flex flex-col gap-3">
      {/* Screen Frame - Clean, sharp technical border instead of soft glowing cards */}
      <div className="relative flex flex-col rounded border border-border bg-[#05070a] overflow-hidden">
        {/* Top HUD Telemetry Overlay */}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2">
          <span className="flex items-center gap-2 rounded bg-black/80 px-2.5 py-1 font-mono text-[11px] text-red-400 border border-red-500/30">
            <span className="size-1.5 bg-red-500 animate-pulse" />
            REC // CAM-04
          </span>
        </div>

        <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
          <span className="flex items-center gap-1.5 rounded bg-black/80 px-2.5 py-1 font-mono text-[11px] text-primary border border-primary/30">
            AI_TRACKING: ACTIVE
          </span>
        </div>

        {/* Video Screen Viewport */}
        <div className="relative aspect-video w-full flex items-center justify-center bg-[#020305]">
          {/* Subtle engineering grid blueprint overlay */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#1a223015_1px,transparent_1px),linear-gradient(to_bottom,#1a223015_1px,transparent_1px)] bg-[size:3rem_3rem]" />

          {/* Central Play Action Trigger */}
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="group relative flex size-14 items-center justify-center rounded bg-primary/10 border border-primary/40 text-primary transition-all hover:bg-primary/20 hover:border-primary"
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="size-5" /> : <Play className="size-5 translate-x-px" />}
          </button>

          {/* Timestamp Telemetry Footer on Video */}
          <div className="absolute bottom-3 left-3 font-mono text-[11px] text-muted-foreground tracking-widest">
            NODE_ID: 4471-B // FPS: 59.94
          </div>
        </div>

        {/* Bottom Hardware Control Bar */}
        <div className="flex h-11 items-center justify-between px-3 border-t border-border bg-card">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
            </button>
            <span className="font-mono text-xs text-muted-foreground">
              01:14:03 / 02:41:57
            </span>
          </div>

          <div className="flex items-center gap-3 text-muted-foreground">
            <button className="hover:text-foreground transition-colors">
              <Volume2 className="size-4" />
            </button>
            <button className="hover:text-foreground transition-colors">
              <Maximize2 className="size-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Timeline Controls - Sharp container with strict alignment */}
      <div className="flex flex-col gap-2 rounded border border-border bg-card p-3">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] tracking-widest text-muted-foreground uppercase">
            Timeline Index
          </span>
          <span className="font-mono text-[10px] text-muted-foreground">
            {markers.length} EVENTS FLAGGED
          </span>
        </div>

        <div className="relative py-2 cursor-pointer">
          <div className="relative h-1.5 w-full rounded-full bg-secondary overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full bg-primary/50"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <input
            type="range"
            min={0}
            max={totalDuration}
            value={currentTime}
            onChange={(e) => setCurrentTime(Number(e.target.value))}
            className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-20"
          />

          <div
            className="absolute top-1/2 -translate-y-1/2 size-3 -ml-1.5 rounded-full bg-foreground border border-primary pointer-events-none z-10"
            style={{ left: `${progressPercent}%` }}
          />

          {markers.map((marker) => {
            const isActive = activeMarkerId === marker.id
            return (
              <button
                key={marker.id}
                type="button"
                onClick={() => {
                  onScrub(marker.id)
                  setCurrentTime((marker.position / 100) * totalDuration)
                }}
                style={{ left: `${marker.position}%` }}
                className={cn(
                  'absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border transition-all hover:scale-125 z-10',
                  marker.color === 'primary' ? 'bg-primary border-primary' : 'bg-slate-700 border-slate-500',
                  isActive ? 'ring-1 ring-primary ring-offset-2 ring-offset-background scale-125' : 'opacity-60'
                )}
                title={marker.label}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}