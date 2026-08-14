'use client'

import { useState } from 'react'
import { Search, UploadCloud, Loader2, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'

interface QueryBarProps {
  onAnalyze: () => void
  isProcessing: boolean
}

export function QueryBar({ onAnalyze, isProcessing }: QueryBarProps) {
  const [query, setQuery] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)

  return (
    <div className="flex flex-col gap-3 md:flex-row">
      <form
        className="relative flex flex-1 items-center gap-2 rounded-lg border border-border bg-card p-1.5 shadow-[0_0_0_1px_rgba(0,0,0,0.2)] focus-within:ring-2 focus-within:ring-primary/40"
        onSubmit={(e) => {
          e.preventDefault()
          onAnalyze()
        }}
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
          placeholder="e.g., Locate a delivery person carrying a blue backpack"
          className="min-w-0 flex-1 bg-transparent py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
        />
        <button
          type="submit"
          disabled={isProcessing}
          className={cn(
            'flex shrink-0 items-center gap-2 rounded-md bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground shadow-[0_0_20px_-4px_var(--primary)] transition-all hover:brightness-110 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60',
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

      <label
        className={cn(
          'flex shrink-0 cursor-pointer items-center gap-3 rounded-lg border px-4 py-2.5 text-sm transition-colors md:w-72',
          isDragging
            ? 'border-accent bg-accent/10 text-accent'
            : 'border-dashed border-border bg-card text-muted-foreground hover:border-accent/50 hover:text-foreground',
        )}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          const file = e.dataTransfer.files?.[0]
          if (file) setFileName(file.name)
        }}
      >
        <UploadCloud className="size-4 shrink-0" aria-hidden="true" />
        <span className="min-w-0 flex-1 truncate">
          {fileName ?? 'Upload Footage (MP4/AVI)'}
        </span>
        <input
          type="file"
          accept="video/mp4,video/avi,video/x-msvideo"
          className="sr-only"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) setFileName(file.name)
          }}
        />
      </label>
    </div>
  )
}
