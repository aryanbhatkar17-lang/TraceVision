'use client'

import { AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Clip {
  id: string
  timestampStart: string
  timestampEnd: string
  type: string
  description: string
  thumbnail: string
}

interface AuditResultsProps {
  clips: Clip[]
  isProcessing: boolean
  activeClipId: string
  onSelect: (id: string) => void
}

export function AuditResults({ clips, isProcessing, activeClipId, onSelect }: AuditResultsProps) {
  return (
    <div className="flex h-full flex-col rounded-xl border border-border bg-card overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-border px-4 bg-card/60 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <span className="text-primary">
            <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
          </span>
          <span className="font-semibold tracking-wider text-sm">Audit Results</span>
        </div>
        <span className="font-mono text-xs text-muted-foreground">
          {clips.length} matches
        </span>
      </div>

      {/* Content List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 scrollbar-thin">
        {isProcessing ? (
          <div className="flex h-32 items-center justify-center text-xs font-mono text-primary animate-pulse">
            ANALYZING FOOTAGE...
          </div>
        ) : clips.length === 0 ? (
          <div className="flex h-32 flex-col items-center justify-center gap-2 text-center text-xs text-muted-foreground">
            <AlertCircle className="size-4 text-muted-foreground" />
            <span>No matching audit clips found.</span>
          </div>
        ) : (
          clips.map((clip) => {
            const isSelected = activeClipId === clip.id
            return (
              <div
                key={clip.id}
                onClick={() => onSelect(clip.id)}
                className={cn(
                  'group relative flex gap-3.5 p-3 rounded-xl border transition-all cursor-pointer backdrop-blur-sm',
                  isSelected
                    ? 'bg-primary/5 border-primary shadow-[0_0_20px_-5px_rgba(16,185,129,0.3)]'
                    : 'bg-card/40 border-border hover:border-primary/50 hover:bg-card/80'
                )}
              >
                {/* Thumbnail Preview */}
                <div className="relative aspect-square w-20 shrink-0 rounded-lg bg-black overflow-hidden border border-border/80">
                  <img
                    src={clip.thumbnail}
                    alt={clip.description}
                    className="h-full w-full object-cover grayscale contrast-125"
                  />
                  <div className="absolute inset-0 bg-primary/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>

                {/* Metadata Details */}
                <div className="flex flex-1 flex-col justify-between min-w-0">
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono text-xs tracking-wider text-primary font-bold">
                      {clip.timestampStart} – {clip.timestampEnd}
                    </span>
                    <span className={cn(
                      'px-2 py-0.5 rounded-full font-mono text-[10px] tracking-wider uppercase border',
                      clip.type === 'PERSON' ? 'bg-blue-950/40 text-blue-400 border-blue-800/50' :
                      clip.type === 'VEHICLE' ? 'bg-amber-950/40 text-amber-400 border-amber-800/50' :
                      'bg-red-950/40 text-red-400 border-red-800/50'
                    )}>
                      {clip.type}
                    </span>
                  </div>

                  <p className="text-xs text-muted-foreground group-hover:text-foreground line-clamp-3 leading-relaxed transition-colors">
                    {clip.description}
                  </p>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}