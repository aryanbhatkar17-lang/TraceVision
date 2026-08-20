'use client'

import { useState, useRef, FormEvent, DragEvent } from 'react'
import { Search, UploadCloud, Loader2, Sparkles, FileVideo } from 'lucide-react'
import { cn } from '@/lib/utils'

interface QueryBarProps {
  onAnalyze: (query: string) => void
  onFileUpload: (file: File) => void
  onCancel?: () => void
  isProcessing: boolean
  hasVideo: boolean
  currentFileName?: string | null
}

export function QueryBar({
  onAnalyze,
  onFileUpload,
  onCancel,
  isProcessing,
  hasVideo,
  currentFileName,
}: QueryBarProps) {
  const [query, setQuery] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    onAnalyze(query.trim())
  }

  const handleDragOver = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) {
      onFileUpload(file)
    }
  }

  return (
    <div className="flex flex-col gap-3 md:flex-row items-stretch">
      {/* Search / Analysis Query Bar */}
      <form
        className="relative flex flex-1 items-center gap-2 rounded-lg border border-border bg-card p-1.5 shadow-[0_0_0_1px_rgba(0,0,0,0.2)] focus-within:ring-2 focus-within:ring-primary/40"
        onSubmit={handleSubmit}
      >
        <Search
          className="ml-2.5 size-4 shrink-0 text-muted-foreground"
          aria-hidden="true"
        />
        <label htmlFor="query-input" className="sr-only">
          Describe what to locate in the footage
        </label>
        <input
          id="query-input"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={
            hasVideo
              ? 'e.g., Locate delivery person, red hatchback, lingering subject, or bus'
              : 'Upload CCTV footage first, then enter audit query'
          }
          className="min-w-0 flex-1 bg-transparent py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
        />

        {isProcessing && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="flex shrink-0 items-center gap-1 rounded px-2.5 py-1.5 text-xs font-mono text-red-400 hover:bg-red-950/40 border border-red-900/40 transition-colors"
          >
            Cancel
          </button>
        )}

        <button
          type="submit"
          disabled={isProcessing || !query.trim() || !hasVideo}
          className={cn(
            'flex shrink-0 items-center gap-2 rounded-md bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground shadow-[0_0_20px_-4px_var(--primary)] transition-all hover:brightness-110 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40',
          )}
        >
          {isProcessing ? (
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          ) : (
            <Sparkles className="size-4" aria-hidden="true" />
          )}
          <span>{isProcessing ? 'Analyzing…' : 'Analyze'}</span>
        </button>
      </form>

      {/* Upload Footage Action */}
      <label
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          'flex shrink-0 cursor-pointer items-center justify-between gap-3 rounded-lg border px-4 py-2.5 text-sm transition-colors md:w-80 select-none',
          isDragging
            ? 'border-primary bg-primary/10 text-primary'
            : currentFileName
            ? 'border-primary/40 bg-card text-foreground hover:border-primary/60'
            : 'border-dashed border-border bg-card text-muted-foreground hover:border-primary/50 hover:text-foreground',
        )}
      >
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          {currentFileName ? (
            <FileVideo className="size-4 shrink-0 text-primary" />
          ) : (
            <UploadCloud className="size-4 shrink-0 text-muted-foreground" />
          )}
          <span className="truncate font-mono text-xs">
            {currentFileName || 'Upload Footage (MP4/AVI/WEBM)'}
          </span>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/avi,video/x-msvideo,video/webm,video/quicktime,video/x-matroska"
          className="sr-only"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) onFileUpload(file)
          }}
        />
      </label>
    </div>
  )
}
