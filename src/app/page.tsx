import Link from 'next/link'
import { Shield, Sparkles, Video, Search, Moon, ArrowRight } from 'lucide-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col justify-between">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="size-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
              <Shield className="size-4" />
            </div>
            <span className="font-bold tracking-wider text-sm">TraceVision</span>
          </div>
          <Link
            href="/dashboard"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-semibold text-sm hover:brightness-110 transition-all shadow-[0_0_15px_-3px_var(--color-primary)]"
          >
            <span>Launch Console</span>
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-5xl mx-auto px-6 py-16 flex-1 flex flex-col justify-center text-center">
        <div className="inline-flex items-center gap-2 self-center px-3.5 py-1.5 rounded-full bg-secondary text-secondary-foreground border border-border text-xs font-mono mb-6">
          <Sparkles className="size-3.5 text-primary" />
          <span>SIH 2026 Innovation Project</span>
        </div>

        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight max-w-3xl mx-auto leading-tight">
          AI-Powered CCTV Video Summarization & Search
        </h1>

        <p className="mt-6 text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
          Locate specific visual events in hours of security footage using natural language queries. Built for low-light, low-resolution, and heavy video streams.
        </p>

        <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/dashboard"
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3.5 rounded-lg bg-primary text-primary-foreground font-bold text-base hover:brightness-110 transition-all shadow-[0_0_20px_-3px_var(--color-primary)]"
          >
            <span>Open Audit Dashboard</span>
            <ArrowRight className="size-5" />
          </Link>
        </div>

        {/* Core Capability Highlights */}
        <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
          <div className="p-6 rounded-xl border border-border bg-card shadow-sm space-y-3">
            <div className="size-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
              <Search className="size-5" />
            </div>
            <h3 className="font-bold text-base">Natural Language Search</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Type queries like &quot;red hatchback in morning&quot; or &quot;person wearing yellow hoodie&quot; to fetch exact timestamps instantly.
            </p>
          </div>

          <div className="p-6 rounded-xl border border-border bg-card shadow-sm space-y-3">
            <div className="size-10 rounded-lg bg-accent flex items-center justify-center text-accent-foreground">
              <Moon className="size-5" />
            </div>
            <h3 className="font-bold text-base">Low-Light Enhancement</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Optimized for night-time footage and blurry feeds using feature-preserving zero-reference frame pre-processing.
            </p>
          </div>

          <div className="p-6 rounded-xl border border-border bg-card shadow-sm space-y-3">
            <div className="size-10 rounded-lg bg-secondary flex items-center justify-center text-secondary-foreground">
              <Video className="size-5" />
            </div>
            <h3 className="font-bold text-base">Heavy File Pipelines</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Processes multi-gigabyte video files efficiently via pre-sampled vector embeddings and fast frame scrubbing.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-6 text-center text-xs font-mono text-muted-foreground">
        TraceVision CCTV Audit Engine — Built for Smart India Hackathon 2026
      </footer>
    </div>
  )
}