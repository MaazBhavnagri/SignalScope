// Typed API client for the SignalScope backend.

export type Verdict = 'real' | 'ai_generated'
export type Band = 'likely_ai' | 'leaning_ai' | 'uncertain' | 'leaning_real' | 'likely_real'

export interface Cue {
  id: string; name: string; value: number; typical_range: [number, number]; fired: boolean; side: 'high' | 'low' | null; strength: number; finding: string
}
export interface RegionCell { region: string; row: number; col: number; share: number; peak: number }
export interface Regions { cells: RegionCell[]; top_regions: string[]; top2_share: number; concentration: 'concentrated' | 'moderate' | 'diffuse'; hot_bbox: [number, number, number, number] | null; hot_area_fraction: number }
export interface Attribution {
  generator: string; family: string; confidence: number; family_confidence: number; distribution: Record<string, number>; family_distribution: Record<string, number>
  p_real_head: number; known_generators: string[]; note: string; applicable?: boolean
}
export interface Provenance {
  signal: 'supports_ai' | 'supports_real' | 'neutral' | 'stripped'; reasons: string[]; camera: Record<string, unknown>; software: string | null; editor_hits: string[]
  ai_markers: string[]; c2pa: { present: boolean; claim_generator: string | null; note: string | null }; gps_present: boolean; jpeg_quality_estimate: number | null
  format: string; mode: string; exif_field_count: number; text_chunks: Record<string, string>
}
export interface Combined { combined_probability: number; rule: string; conflict: boolean }
export interface CaptionConsistency { caption?: string; similarity?: number; rank_vs_distractors?: number; n_distractors?: number; band: 'consistent' | 'weak' | 'inconsistent' | null; sentence?: string; model?: string; error?: string }
export interface Explanation { band: Band; label: string; summary: string; sentences: string[]; fired_cues: string[] }
export interface RobustnessItem { id: string; label: string; probability_ai: number; delta: number; verdict: Verdict; flipped: boolean }
export interface Robustness { base_probability_ai: number; items: RobustnessItem[]; flips: number; max_shift: number; stability: 'stable' | 'moderate' | 'fragile'; note: string }

export interface AnalysisSummary {
  id: string; created_at: string; batch_id: string | null; filename: string; width: number; height: number; size_bytes: number
  verdict: Verdict; label: string; band: Band; probability_ai: number; confidence: number; threshold: number
  attribution_family: string | null; provenance_signal: string | null; fired_cues: string[]; review_decision: string | null
  has_robustness: boolean; stability: string | null; latency_ms: number | null; image_url: string; heatmap_url: string | null
}
export interface Analysis extends AnalysisSummary {
  content_type: string | null; sha256: string; probability_ai_uncalibrated: number
  attribution: Attribution | null; cues: Cue[]; regions: Regions | null; provenance: Provenance | null; combined: Combined | null
  caption: string | null; caption_consistency: CaptionConsistency | null; explanation: Explanation; robustness: Robustness | null
  timing_ms: { model: number; total: number }; model: Record<string, unknown>; review_note: string | null; reviewed_at: string | null
}
export interface BatchInfo { id: string; name: string | null; created_at: string; n_items: number; n_ai: number; n_real: number; n_uncertain: number; total_ms: number }
export interface BatchResult { batch: BatchInfo; items: AnalysisSummary[]; errors: { filename: string; detail: unknown }[] }
export interface ListResult { total: number; items: AnalysisSummary[]; limit: number; offset: number }
export interface Health { status: string; model: { loaded: boolean; error: string | null; checkpoint: string; device: string | null; threshold: number | null; temperature: number | null; epoch: number | null; val_auc: number | null }; time: string }
export interface Stats { total: number; by_verdict: Record<string, number>; by_band: Record<string, number>; by_review: Record<string, number>; mean_probability_ai: number; per_day: Record<string, { real: number; ai_generated: number }>; latency_ms: { median: number | null; n: number }; families: Record<string, number>; batches: number }

export class ApiError extends Error {
  status: number; code: string
  constructor(status: number, code: string, message: string) { super(message); this.status = status; this.code = code }
}

async function handle<T>(r: Response): Promise<T> {
  if (r.status === 204) return undefined as T
  const text = await r.text()
  let data: unknown = null
  try { data = text ? JSON.parse(text) : null } catch { /* non-JSON */ }
  if (!r.ok) {
    const d = (data as { detail?: { error?: string; message?: string } | string } | null)?.detail
    const code = typeof d === 'object' && d ? d.error ?? 'error' : 'error'
    const msg = typeof d === 'object' && d ? d.message ?? r.statusText : typeof d === 'string' ? d : r.statusText
    throw new ApiError(r.status, code, msg || `Request failed (${r.status})`)
  }
  return data as T
}

export const api = {
  health: () => fetch('/api/health').then(r => handle<Health>(r)),
  stats: () => fetch('/api/stats').then(r => handle<Stats>(r)),
  modelCard: () => fetch('/api/model').then(r => handle<Record<string, any>>(r)),
  analyze: (file: File, caption?: string, heatmap = true) => {
    const fd = new FormData(); fd.append('file', file); if (caption) fd.append('caption', caption); fd.append('heatmap', String(heatmap))
    return fetch('/api/analyze', { method: 'POST', body: fd }).then(r => handle<Analysis>(r))
  },
  analyzeBatch: (files: File[], name?: string) => {
    const fd = new FormData(); files.forEach(f => fd.append('files', f)); if (name) fd.append('name', name)
    return fetch('/api/analyze/batch', { method: 'POST', body: fd }).then(r => handle<BatchResult>(r))
  },
  list: (params: Record<string, string | number | undefined>) => {
    const q = new URLSearchParams(); Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== '') q.set(k, String(v)) })
    return fetch(`/api/analyses?${q}`).then(r => handle<ListResult>(r))
  },
  get: (id: string) => fetch(`/api/analyses/${id}`).then(r => handle<Analysis>(r)),
  remove: (id: string) => fetch(`/api/analyses/${id}`, { method: 'DELETE' }).then(r => handle<void>(r)),
  review: (id: string, decision: string | null, note: string | null) =>
    fetch(`/api/analyses/${id}/review`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision, note }) }).then(r => handle<Analysis>(r)),
  robustness: (id: string) => fetch(`/api/analyses/${id}/robustness`, { method: 'POST' }).then(r => handle<Robustness>(r)),
  batches: () => fetch('/api/batches').then(r => handle<{ items: BatchInfo[] }>(r)),
  batch: (id: string) => fetch(`/api/batches/${id}`).then(r => handle<{ batch: BatchInfo; items: AnalysisSummary[] }>(r)),
  reportUrl: (id: string, format: 'html' | 'json' = 'html') => `/api/analyses/${id}/report?format=${format}`,
}

export const BAND_LABEL: Record<Band, string> = {
  likely_ai: 'Likely AI-generated', leaning_ai: 'Possibly AI-generated', uncertain: 'Inconclusive', leaning_real: 'Possibly real', likely_real: 'Likely real photo',
}
export function tone(band: Band): 'ai' | 'real' | 'unsure' {
  if (band === 'uncertain') return 'unsure'
  return band.endsWith('_ai') ? 'ai' : 'real'
}
export const pct = (p: number, d = 0) => `${(p * 100).toFixed(d)}%`
export const fmtBytes = (b: number) => b > 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${(b / 1024).toFixed(0)} KB`
export const fmtDate = (s: string) => new Date(s).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
