'use client'

import { useState } from 'react'
import { Sidebar } from './sidebar'
import { Topbar } from './topbar'
import { QueryBar } from './query-bar'
import { VideoPlayer } from './video-player'
import { AuditResults } from './audit-results'

const MARKERS = [
  { id: 'clip-1', position: 15, label: 'Delivery person entering frame', color: 'primary' as const },
  { id: 'clip-2', position: 38, label: 'Red hatchback crossing junction', color: 'accent' as const },
  { id: 'clip-3', position: 65, label: 'Individual lingering near garage', color: 'accent' as const },
  { id: 'clip-4', position: 88, label: 'Subject re-entering east lobby', color: 'primary' as const },
]

const CLIPS = [
  {
    id: 'clip-1',
    timestampStart: '01:14:05',
    timestampEnd: '01:14:10',
    type: 'PERSON',
    description: 'Delivery person carrying a blue backpack enters frame from the north walkway, pauses briefly at the loading dock.',
    thumbnail: 'https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=300&auto=format&fit=crop&q=80',
  },
  {
    id: 'clip-2',
    timestampStart: '01:22:41',
    timestampEnd: '01:22:47',
    type: 'VEHICLE',
    description: 'Red hatchback detected crossing the junction against the pedestrian signal, moderate confidence match.',
    thumbnail: 'https://images.unsplash.com/photo-1506521781263-d8422e82f27a?w=300&auto=format&fit=crop&q=80',
  },
  {
    id: 'clip-3',
    timestampStart: '01:47:12',
    timestampEnd: '01:48:03',
    type: 'ANOMALY',
    description: 'Single individual lingers near the garage exit for 51 seconds with no vehicle interaction observed.',
    thumbnail: 'https://images.unsplash.com/photo-1590381105924-c72589b9ef3f?w=300&auto=format&fit=crop&q=80',
  },
  {
    id: 'clip-4',
    timestampStart: '02:03:56',
    timestampEnd: '02:04:02',
    type: 'PERSON',
    description: 'Same subject from clip 01 re-enters through the east lobby doors carrying the same blue backpack.',
    thumbnail: 'https://images.unsplash.com/photo-1497366216548-37526070297c?w=300&auto=format&fit=crop&q=80',
  },
]

export function Dashboard() {
  const [activeId, setActiveId] = useState('clip-1')
  const [isProcessing, setIsProcessing] = useState(false)
  const [hasResults, setHasResults] = useState(true)

  const handleAnalyze = (query?: string) => {
    setIsProcessing(true)
    setTimeout(() => {
      setIsProcessing(false)
      setHasResults(true)
    }, 800)
  }

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <Sidebar />
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        <Topbar />
        
        <main className="flex-1 overflow-y-auto p-4 lg:p-6 space-y-4">
          <QueryBar onAnalyze={handleAnalyze} isProcessing={isProcessing} />

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
            <VideoPlayer
              markers={MARKERS}
              activeMarkerId={activeId}
              onScrub={setActiveId}
            />

            <div className="h-[450px] lg:h-[calc(100vh-14rem)]">
              <AuditResults
                clips={hasResults ? CLIPS : []}
                isProcessing={isProcessing}
                activeClipId={activeId}
                onSelect={setActiveId}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}