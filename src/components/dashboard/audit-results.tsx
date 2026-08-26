'use client'

import { AlertCircle, Loader2, XCircle, ArrowUpRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { AuditMatch, AnalysisProgress } from '@/types/audit'

interface AuditResultsProps {
  matches: AuditMatch[]
  isProcessing: boolean
  progress?: AnalysisProgress | null
  activeMatchId: string | null
  onSelect: (id: string, startSeconds: number) => void
  onCancel?: () => void
}

export function AuditResults({
  matches,
  isProcessing,
  progress,
  activeMatchId,
  onSelect,
  onCancel,
}: AuditResultsProps) {
  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-card overflow-hidden shadow-2xl">
      {/* Panel Header */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-border px-4 bg-card/70 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <span className="text-blue-600">
            <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
          </span>
          <span className="font-semibold tracking-wider text-sm">Audit Results</span>
        </div>
        <span className="font-mono text-xs text-muted-foreground">
          {`${matches.length} ${matches.length === 1 ? 'match' : 'matches'}`}
        </span>
      </div>

      {/* Panel Body */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 scrollbar-thin">
        {isProcessing ? (
          /* Real-Time Processing Progress - Blue Themed */
          <div className="flex flex-col items-center justify-center p-6 space-y-4 rounded-lg border border-blue-500/20 bg-blue-500/5 my-auto text-center animate-in fade-in duration-300">
            <div className="relative flex size-12 items-center justify-center rounded-full bg-blue-500/10 border border-blue-500/40">
              <Loader2 className="size-6 text-blue-600 animate-spin" />
            </div>

            <div className="space-y-1.5 w-full max-w-xs">
              <div className="flex items-center justify-between text-xs font-mono text-blue-600 font-bold">
                <span>
                  {progress?.status === 'uploading'
                    ? 'UPLOADING'
                    : progress?.status === 'extracting'
                      ? 'EXTRACTING'
                      : 'ANALYZING'}
                </span>
                <span>{progress?.progress ?? 35}%</span>
              </div>

              {/* Progress Bar */}
              <div className="h-2 w-full rounded-full bg-secondary overflow-hidden border border-border/50">
                <div
                  className="h-full bg-blue-600 transition-all duration-300 shadow-[0_0_10px_rgba(37,99,235,0.4)]"
                  style={{ width: `${progress?.progress ?? 35}%` }}
                />
              </div>

              <p className="text-xs text-muted-foreground font-mono pt-1 leading-relaxed">
                {progress?.message || 'Processing video chunks & extracting keyframes...'}
              </p>
            </div>

            {onCancel && (
              <button
                type="button"
                onClick={onCancel}
                className="flex items-center gap-1.5 rounded px-2.5 py-1 font-mono text-[11px] text-red-400 hover:text-red-300 hover:bg-red-950/30 border border-red-900/40 transition-colors"
              >
                <XCircle className="size-3.5" />
                <span>Cancel Audit</span>
              </button>
            )}
          </div>
        ) : matches.length === 0 ? (
          /* Empty State */
          <div className="flex h-64 flex-col items-center justify-center p-6 text-center space-y-3">
            <div className="flex size-12 items-center justify-center rounded-xl bg-secondary/80 border border-border/80 text-muted-foreground">
              <AlertCircle className="size-6 text-muted-foreground" />
            </div>
            <div className="space-y-1 max-w-xs">
              <p className="text-xs font-medium text-foreground">
                No audit queries executed.
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Upload footage and submit an analysis query.
              </p>
            </div>
          </div>
        ) : (
          /* Simplified Result Cards (Tags and Confidence Removed) */
          matches.map((match) => {
            const isSelected = activeMatchId === match.id

            return (
              <div
                key={match.id}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(match.id, match.start_seconds)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onSelect(match.id, match.start_seconds)
                  }
                }}
                className={cn(
                  'group relative flex flex-col gap-2 p-3.5 rounded-lg border transition-all cursor-pointer select-none text-left',
                  isSelected
                    ? 'bg-blue-500/10 border-blue-500 shadow-[0_0_20px_-5px_rgba(37,99,235,0.3)] ring-1 ring-blue-500/40'
                    : 'bg-card/50 border-border hover:border-blue-500/50 hover:bg-card/90'
                )}
              >
                {/* Header Row: Timestamp */}
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-xs font-bold tracking-wider text-blue-600">
                      {match.start_time} – {match.end_time}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground/60">
                      ({Math.round(match.start_seconds)}s)
                    </span>
                  </div>
                </div>

                {/* Event Description */}
                <p className="text-xs text-muted-foreground group-hover:text-foreground leading-relaxed transition-colors">
                  {match.description}
                </p>

                {/* Footer Metadata & Seek Prompt */}
                <div className="flex items-center justify-between pt-1 border-t border-border/40 text-[10px] font-mono text-muted-foreground/70">
                  <span className="flex items-center gap-1">
                    {match.chunk_id && (
                      <span className="text-muted-foreground/40">{match.chunk_id}</span>
                    )}
                  </span>
                  <span className="flex items-center gap-0.5 text-blue-600 opacity-0 group-hover:opacity-100 transition-opacity">
                    Seek to frame <ArrowUpRight className="size-3" />
                  </span>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}