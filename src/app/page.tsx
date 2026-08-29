'use client'

import Link from 'next/link'
import Image from 'next/image'
import { Search, FileText, ArrowRight, Video, Lock, ShieldCheck } from 'lucide-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#f0f4f8] text-slate-900 flex flex-col justify-between font-sans selection:bg-blue-100 selection:text-blue-900">

      {/* 1. Full-Width Edge-to-Edge Glassmorphism Floating Navbar */}
      <div className="sticky top-0 z-50 pt-4 px-4 sm:px-8 pointer-events-none">
        <header className="w-full h-14 px-6 sm:px-10 rounded-2xl bg-white/70 backdrop-blur-xl backdrop-saturate-150 border border-white/80 shadow-[0_10px_30px_-5px_rgba(15,23,42,0.12),inset_0_1px_1px_rgba(255,255,255,0.9)] flex items-center justify-between transition-all pointer-events-auto">

          {/* Left Corner: Logo & Identity */}
          <Link href="/" className="flex items-center space-x-3 group">
            <div className="relative w-7 h-7 flex items-center justify-center rounded-lg overflow-hidden shrink-0 shadow-[inset_0_1px_1px_rgba(255,255,255,0.6)]">
              <Image
                src="/tracevision-icon.svg"
                alt="TraceVision Icon"
                width={28}
                height={28}
                className="object-contain transition-transform duration-200 group-hover:scale-105"
                priority
              />
            </div>
            <span className="text-sm font-bold tracking-wider uppercase text-slate-800 group-hover:text-blue-700 transition-colors">
              TraceVision
            </span>
          </Link>

          {/* Right Corner: Glass Badging */}
          <div className="flex items-center">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium text-slate-700 bg-white/60 backdrop-blur-md border border-white/90 shadow-[0_2px_8px_rgba(0,0,0,0.04),inset_0_1px_1px_rgba(255,255,255,0.9)]">
              <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
              Investigation Portal
            </span>
          </div>
        </header>
      </div>

      {/* 2. Main Hero Section */}
      <main className="max-w-5xl w-full mx-auto px-6 py-12 lg:py-16 flex-1 flex flex-col justify-center text-center">

        {/* Main Headline */}
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 max-w-4xl mx-auto leading-tight">
          AI-Powered CCTV Video Summarization & Search
        </h1>

        {/* Subtitle */}
        <p className="mt-5 text-base sm:text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Locate specific visual events in hours of security footage using plain language queries. Built for low-light, low-resolution, and heavy surveillance streams.
        </p>

        {/* Primary Action Button */}
        <div className="mt-8 flex items-center justify-center">
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-blue-700 hover:bg-blue-800 text-white font-bold text-sm shadow-md hover:shadow-lg transition-all"
          >
            <span>Start Video Investigation</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        {/* Standard Investigation Workflow (3 Columns) */}
        <div className="mt-16 text-left">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-4 text-center">
            Standard Investigation Workflow
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Step 1 */}
            <div className="p-6 rounded-2xl border border-white/80 bg-white/80 backdrop-blur-md shadow-[0_4px_20px_rgba(0,0,0,0.03)] space-y-2">
              <div className="flex items-center gap-2 text-blue-800 font-bold text-sm">
                <div className="p-2 bg-blue-50 border border-blue-100 rounded-lg text-blue-700">
                  <Video className="w-4 h-4" />
                </div>
                Step 1: Load Video
              </div>
              <p className="text-xs text-slate-600 leading-relaxed pt-1">
                Select or drag-and-drop the CCTV video clip directly from your computer (MP4, AVI, or MKV).
              </p>
            </div>

            {/* Step 2 */}
            <div className="p-6 rounded-2xl border border-white/80 bg-white/80 backdrop-blur-md shadow-[0_4px_20px_rgba(0,0,0,0.03)] space-y-2">
              <div className="flex items-center gap-2 text-blue-800 font-bold text-sm">
                <div className="p-2 bg-blue-50 border border-blue-100 rounded-lg text-blue-700">
                  <Search className="w-4 h-4" />
                </div>
                Step 2: Type What You Seek
              </div>
              <p className="text-xs text-slate-600 leading-relaxed pt-1">
                Enter descriptions like <span className="font-semibold text-slate-800">&quot;red motorbike&quot;</span> or <span className="font-semibold text-slate-800">&quot;person in dark hoodie&quot;</span>.
              </p>
            </div>

            {/* Step 3 */}
            <div className="p-6 rounded-2xl border border-white/80 bg-white/80 backdrop-blur-md shadow-[0_4px_20px_rgba(0,0,0,0.03)] space-y-2">
              <div className="flex items-center gap-2 text-blue-800 font-bold text-sm">
                <div className="p-2 bg-blue-50 border border-blue-100 rounded-lg text-blue-700">
                  <FileText className="w-4 h-4" />
                </div>
                Step 3: Review Evidence
              </div>
              <p className="text-xs text-slate-600 leading-relaxed pt-1">
                Click on matched timestamps to play key moments and inspect detections with exact time markers.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* 3. Official Departmental Footer */}
      <footer className="border-t border-slate-200/80 bg-white/60 backdrop-blur-md py-4 px-6 text-center text-xs text-slate-500">
        <div className="flex items-center justify-center gap-1.5">
          <Lock className="w-3.5 h-3.5 text-slate-400" />
          <span>Internal Police & Forensic Investigation Portal • Authorized Access Only</span>
        </div>
      </footer>

    </div>
  )
}