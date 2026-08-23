export type AuditCategory = 'PERSON' | 'VEHICLE' | 'ANOMALY' | 'SECURITY' | 'OBJECT' | 'MOTION'

export interface AuditMatch {
  id: string
  start_time: string
  end_time: string
  start_seconds: number
  end_seconds: number
  category: AuditCategory | string
  description: string
  confidence?: number
  chunk_id?: string
}

export interface AuditResponse {
  matches: AuditMatch[]
  total_chunks?: number
  video_duration?: number
  query?: string
}

export interface TimelineMarker {
  id: string
  position: number // Percentage 0 - 100
  seconds: number
  label: string
  color: 'primary' | 'accent' | 'destructive'
  category: string
}

export interface AnalysisProgress {
  status: 'idle' | 'compressing' | 'uploading' | 'extracting' | 'analyzing' | 'aggregating' | 'completed' | 'error' | 'cancelled'
  progress: number // Percentage 0 - 100
  message: string
  currentSegment?: number
  totalSegments?: number
}

export interface ChunkMapping {
  chunk_id: string
  start_second: number
  end_second: number
  original_filename: string
  frame_count?: number
}
