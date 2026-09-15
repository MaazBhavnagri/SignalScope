import { useEffect, useState } from 'react'
import { api, pct, tone, type AnalysisSummary, type BatchInfo, type BatchResult } from '../api'
import Dropzone from '../components/Dropzone'

interface Props { notify: (k: 'ok' | 'err', t: string) => void; openCase: (id: string) => void }

export function CaseCard({ a, onOpen, selected, onSelect }: { a: AnalysisSummary; onOpen: () => void; selected?: boolean; onSelect?: () => void }) {
  const t = tone(a.band)
  return (
    <div
      className={`card ${selected ? 'selected' : ''}`}
      onClick={onOpen}
      role="button"
      tabIndex={0}
      onKeyDown={e => { if (e.key === 'Enter') onOpen() }}
    >
      <div className="thumb">
        <img src={a.image_url} alt={a.filename} loading="lazy" />
        <span className="p">{pct(a.probability_ai)}</span>
        <span className={`bandline bg-${t}`} />
      </div>
      <div className="body">
        <div className="fn" title={a.filename}>{a.filename}</div>
        <div className="meta">
          <span className={`chip ${t}`}>
            {a.band === 'uncertain' ? 'inconclusive' : t === 'ai' ? 'AI' : 'real'}
          </span>
          {onSelect && (
            <label className="small muted" onClick={e => e.stopPropagation()} style={{ cursor: 'pointer' }}>
              <input type="checkbox" checked={!!selected} onChange={onSelect} /> compare
            </label>
          )}
        </div>
        {(a.attribution_family || a.review_decision || a.stability) && (
          <div className="row small muted" style={{ marginTop: 6, gap: 5, flexWrap: 'wrap' }}>
            {a.attribution_family && a.verdict === 'ai_generated' && <span className="mono">{a.attribution_family}</span>}
            {a.stability && <span className="mono">· {a.stability}</span>}
            {a.review_decision && <span className="mono">· {a.review_decision.replace('_', ' ')}</span>}
          </div>
        )}
      </div>
    </div>
  )
}

export default function BatchView({ notify, openCase }: Props) {
  const [queue, setQueue] = useState<File[]>([])
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<BatchResult | null>(null)
  const [history, setHistory] = useState<BatchInfo[]>([])
  const [filter, setFilter] = useState<'all' | 'ai' | 'real' | 'unsure'>('all')
  const [sort, setSort] = useState<'p_desc' | 'p_asc' | 'name'>('p_desc')

  const loadHistory = () => api.batches().then(r => setHistory(r.items)).catch(() => undefined)
  useEffect(() => { loadHistory() }, [])

  const run = async () => {
    if (!queue.length) return
    setBusy(true)
    try {
      const r = await api.analyzeBatch(queue, name.trim() || undefined)
      setResult(r); setQueue([]); loadHistory()
      notify('ok', `Batch complete: ${r.batch.n_items} images in ${(r.batch.total_ms / 1000).toFixed(1)}s`)
      if (r.errors.length) notify('err', `${r.errors.length} file(s) were rejected`)
    } catch (e) { notify('err', (e as Error).message) } finally { setBusy(false) }
  }

  const openBatch = async (id: string) => {
    try { const r = await api.batch(id); setResult({ ...r, errors: [] }) }
    catch (e) { notify('err', (e as Error).message) }
  }

  const items = (result?.items ?? [])
    .filter(a => filter === 'all' || tone(a.band) === filter)
    .sort((x, y) =>
      sort === 'name'   ? x.filename.localeCompare(y.filename) :
      sort === 'p_desc' ? y.probability_ai - x.probability_ai  :
      x.probability_ai - y.probability_ai
    )

  return (
    <div className="stack fade-in">
      {/* ── Page header ─────────────────────────────────────────── */}
      <div className="page-head entry-1">
        <div>
          <div className="eyebrow">batch scan · module f</div>
          <h1>Scan a folder of images</h1>
          <p>
            Triage up to 50 images in one pass. Each image gets the full analysis and a case file;
            sort by likelihood to review the most suspicious first.
          </p>
        </div>
      </div>

      {/* ── Queue + options ─────────────────────────────────────── */}
      <div className="grid-2 entry-2" style={{ gridTemplateColumns: '1.5fr 1fr', alignItems: 'start' }}>
        <div className="stack">
          <Dropzone
            multiple
            compact={queue.length > 0}
            onFiles={f => setQueue(q => [...q, ...f].slice(0, 50))}
            title={queue.length ? `Add more — ${queue.length} queued` : 'Drop images or a selection'}
            hint="up to 50 images per batch · JPEG, PNG, WebP"
            disabled={busy}
          />
          {queue.length > 0 && (
            <div className="panel">
              <div className="panel-h">
                <h3>Queue <span className="eyebrow">{queue.length} files</span></h3>
                <button className="btn sm ghost" onClick={() => setQueue([])} disabled={busy}>clear all</button>
              </div>
              <div className="panel-b" style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 180, overflowY: 'auto' }}>
                {queue.map((f, i) => (
                  <span key={i} className="chip" title={f.name}>
                    {f.name.length > 28 ? f.name.slice(0, 26) + '…' : f.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="panel" style={{ position: 'sticky', top: 24 }}>
          <div className="panel-h"><h3>Batch options</h3></div>
          <div className="panel-b stack">
            <label className="stack" style={{ gap: 6 }}>
              <span className="eyebrow">batch name <span style={{ opacity: 0.6 }}>optional</span></span>
              <input
                className="input" value={name}
                onChange={e => setName(e.target.value)}
                placeholder="e.g. marketplace listing sweep 09-11"
                maxLength={120}
              />
            </label>
            <button
              className="btn primary"
              style={{ justifyContent: 'center', padding: '13px 16px', fontSize: 14 }}
              disabled={!queue.length || busy}
              onClick={run}
            >
              {busy
                ? <><span className="spinner" /> Scanning {queue.length} images…</>
                : `Scan ${queue.length || ''} image${queue.length === 1 ? '' : 's'}`
              }
            </button>

            {history.length > 0 && (
              <>
                <div className="divider" />
                <div className="eyebrow">recent batches</div>
                <div className="stack" style={{ gap: 5 }}>
                  {history.slice(0, 6).map(b => (
                    <button
                      key={b.id} className="btn sm ghost"
                      style={{ justifyContent: 'space-between' }}
                      onClick={() => openBatch(b.id)}
                    >
                      <span>{b.name || `batch ${b.id.slice(0, 6)}`}</span>
                      <span className="mono muted" style={{ fontSize: 11 }}>
                        {b.n_items} · <span className="t-ai">{b.n_ai}</span>/<span className="t-real">{b.n_real}</span>/<span className="t-unsure">{b.n_uncertain}</span>
                      </span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* ── Results ─────────────────────────────────────────────── */}
      {result && (
        <div className="stack fade-in">
          <div className="strip entry-3">
            <div className="stat">
              <div className="eyebrow">images</div>
              <div className="v">{result.batch.n_items}</div>
            </div>
            <div className="stat">
              <div className="eyebrow">likely / possibly AI</div>
              <div className="v t-ai">{result.batch.n_ai}</div>
            </div>
            <div className="stat">
              <div className="eyebrow">likely / possibly real</div>
              <div className="v t-real">{result.batch.n_real}</div>
            </div>
            <div className="stat">
              <div className="eyebrow">inconclusive</div>
              <div className="v t-unsure">{result.batch.n_uncertain}</div>
            </div>
            <div className="stat">
              <div className="eyebrow">wall time</div>
              <div className="v">{(result.batch.total_ms / 1000).toFixed(1)}<small> s</small></div>
            </div>
          </div>

          {result.errors.length > 0 && (
            <div className="notice err">
              {result.errors.length} file(s) rejected: {result.errors.map(e => `${e.filename} (${(e.detail as { message?: string })?.message ?? 'invalid'})`).join('; ')}
            </div>
          )}

          <div className="entry-4" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
            <div className="tabs" style={{ marginBottom: 0 }}>
              {(['all', 'ai', 'real', 'unsure'] as const).map(f => (
                <button key={f} className={filter === f ? 'active' : ''} onClick={() => setFilter(f)}>
                  {f === 'all' ? 'All' : f === 'ai' ? 'AI-leaning' : f === 'real' ? 'Real-leaning' : 'Inconclusive'}
                </button>
              ))}
            </div>
            <select className="input" style={{ width: 200 }} value={sort} onChange={e => setSort(e.target.value as typeof sort)}>
              <option value="p_desc">Most likely AI first</option>
              <option value="p_asc">Most likely real first</option>
              <option value="name">File name</option>
            </select>
          </div>

          <div className="cards entry-5">
            {items.map(a => <CaseCard key={a.id} a={a} onOpen={() => openCase(a.id)} />)}
          </div>
        </div>
      )}
    </div>
  )
}
