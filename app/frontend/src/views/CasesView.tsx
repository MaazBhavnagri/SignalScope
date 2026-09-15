import { useCallback, useEffect, useState } from 'react'
import { api, fmtDate, pct, tone, type Analysis, type AnalysisSummary, type Stats } from '../api'
import Gauge from '../components/Gauge'
import { CaseCard } from './BatchView'

interface Props { notify: (k: 'ok' | 'err', t: string) => void; openCase: (id: string) => void; compare: string[]; toggleCompare: (id: string) => void; setCompare: (ids: string[]) => void }

const PAGE = 24

function CompareColumn({ a }: { a: Analysis }) {
  const [heat, setHeat] = useState(true)
  return (
    <div className="panel">
      <div className="panel-h">
        <h3 title={a.filename}>{a.filename.length > 34 ? a.filename.slice(0, 32) + '…' : a.filename}</h3>
        <span className={`chip ${tone(a.band)}`}>{a.label}</span>
      </div>
      <div className="panel-b stack">
        <div className="frame" style={{ minHeight: 200 }}>
          <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
            <img src={a.image_url} alt={a.filename} style={{ maxHeight: 320, minWidth: 'min(100%, 280px)' }} />
            {heat && a.heatmap_url && (
              <img src={a.heatmap_url} alt="" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.75 }} />
            )}
          </div>
        </div>
        <label className="small muted row" style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={heat} onChange={e => setHeat(e.target.checked)} /> heat-map
        </label>
        <div className="row" style={{ alignItems: 'center' }}>
          <Gauge p={a.probability_ai} threshold={a.threshold} band={a.band} size={150} />
          <dl className="kv" style={{ flex: 1 }}>
            <dt>calibrated</dt><dd className="mono">{pct(a.probability_ai, 1)}</dd>
            <dt>raw</dt><dd className="mono">{pct(a.probability_ai_uncalibrated, 1)}</dd>
            <dt>regions</dt><dd>{a.regions ? `${a.regions.concentration} · ${a.regions.top_regions.join(', ')}` : '-'}</dd>
            <dt>attribution</dt><dd>{a.verdict === 'ai_generated' && a.attribution ? `${a.attribution.family} (${a.attribution.generator})` : 'n/a'}</dd>
            <dt>metadata</dt><dd>{a.provenance?.signal.replace('_', ' ') ?? '-'}</dd>
            <dt>stability</dt><dd>{a.robustness?.stability ?? 'not tested'}</dd>
            <dt>analysed</dt><dd>{fmtDate(a.created_at)}</dd>
          </dl>
        </div>
        <div>
          <div className="eyebrow" style={{ marginBottom: 8 }}>cues outside real-photo range</div>
          {a.cues.filter(c => c.fired).length ? (
            <div className="bars">
              {a.cues.filter(c => c.fired).map(c => (
                <div className="bar" key={c.id}>
                  <span className="n">{c.name}</span>
                  <div className="track"><div className="fill ai" style={{ width: `${c.strength * 100}%` }} /></div>
                  <span className="v">{Math.round(c.strength * 100)}%</span>
                </div>
              ))}
            </div>
          ) : <div className="small muted">none fired</div>}
        </div>
        <div className="small" style={{ color: 'var(--ink-2)', lineHeight: 1.65 }}>{a.explanation.summary}</div>
      </div>
    </div>
  )
}

export default function CasesView({ notify, openCase, compare, toggleCompare, setCompare }: Props) {
  const [items, setItems] = useState<AnalysisSummary[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [q, setQ] = useState('')
  const [verdict, setVerdict] = useState('')
  const [band, setBand] = useState('')
  const [review, setReview] = useState('')
  const [sort, setSort] = useState<'newest' | 'oldest' | 'p_desc' | 'p_asc'>('newest')
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<Stats | null>(null)
  const [pair, setPair] = useState<Analysis[]>([])

  const load = useCallback(() => {
    setLoading(true)
    api.list({ limit: PAGE, offset, q, verdict, band, review, sort })
      .then(r => { setItems(r.items); setTotal(r.total) })
      .catch(e => notify('err', e.message))
      .finally(() => setLoading(false))
  }, [offset, q, verdict, band, review, sort, notify])

  useEffect(() => { load() }, [load])
  useEffect(() => { api.stats().then(setStats).catch(() => undefined) }, [items])
  useEffect(() => {
    if (compare.length === 2) Promise.all(compare.map(id => api.get(id))).then(setPair).catch(e => notify('err', e.message))
    else setPair([])
  }, [compare, notify])

  const pages = Math.max(1, Math.ceil(total / PAGE))
  const page = Math.floor(offset / PAGE) + 1

  return (
    <div className="stack fade-in">
      {/* ── Page header ─────────────────────────────────────────── */}
      <div className="page-head entry-1">
        <div>
          <div className="eyebrow">case files · analysis history</div>
          <h1>Case files</h1>
          <p>Every scan is stored as a case with its evidence. Filter, review, compare two cases side by side, or export the report.</p>
        </div>
      </div>

      {/* ── Telemetry stats strip ────────────────────────────────── */}
      {stats && stats.total > 0 && (
        <div className="strip entry-2">
          <div className="stat">
            <div className="eyebrow">total cases</div>
            <div className="v">{stats.total}</div>
          </div>
          <div className="stat">
            <div className="eyebrow">flagged AI</div>
            <div className="v t-ai">{stats.by_verdict.ai_generated ?? 0}</div>
          </div>
          <div className="stat">
            <div className="eyebrow">assessed real</div>
            <div className="v t-real">{stats.by_verdict.real ?? 0}</div>
          </div>
          <div className="stat">
            <div className="eyebrow">inconclusive</div>
            <div className="v t-unsure">{stats.by_band.uncertain ?? 0}</div>
          </div>
          <div className="stat">
            <div className="eyebrow">reviewed</div>
            <div className="v">
              {(stats.by_review.confirmed_ai ?? 0) + (stats.by_review.confirmed_real ?? 0)}
              <small> / {stats.total}</small>
            </div>
          </div>
          <div className="stat">
            <div className="eyebrow">median latency</div>
            <div className="v">{stats.latency_ms.median?.toFixed(0) ?? '-'}<small> ms</small></div>
          </div>
        </div>
      )}

      {/* ── Compare pane ────────────────────────────────────────── */}
      {pair.length === 2 && (
        <div className="stack fade-in entry-3">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <h2 style={{ fontFamily: 'var(--serif)', fontSize: 26, fontWeight: 400 }}>Side-by-side comparison</h2>
            <button className="btn sm ghost" onClick={() => setCompare([])}>close comparison</button>
          </div>
          <div className="grid-2">{pair.map(a => <CompareColumn key={a.id} a={a} />)}</div>
        </div>
      )}

      {compare.length === 1 && (
        <div className="notice entry-3">
          One case in the compare tray — tick "compare" on a second case to see them side by side.{' '}
          <button className="btn sm ghost" onClick={() => setCompare([])}>clear</button>
        </div>
      )}

      {/* ── Search + filter panel ────────────────────────────────── */}
      <div className="panel entry-4">
        <div className="panel-h" style={{ flexWrap: 'wrap', gap: 10 }}>
          <div className="row" style={{ flex: 1, minWidth: 0, gap: 8, flexWrap: 'wrap' }}>
            <input
              className="input" style={{ width: 200 }}
              placeholder="search filename"
              value={q}
              onChange={e => { setQ(e.target.value); setOffset(0) }}
            />
            <select className="input" style={{ width: 148 }} value={verdict} onChange={e => { setVerdict(e.target.value); setOffset(0) }}>
              <option value="">any verdict</option>
              <option value="ai_generated">AI-generated</option>
              <option value="real">real</option>
            </select>
            <select className="input" style={{ width: 160 }} value={band} onChange={e => { setBand(e.target.value); setOffset(0) }}>
              <option value="">any band</option>
              <option value="likely_ai">likely AI</option>
              <option value="leaning_ai">possibly AI</option>
              <option value="uncertain">inconclusive</option>
              <option value="leaning_real">possibly real</option>
              <option value="likely_real">likely real</option>
            </select>
            <select className="input" style={{ width: 155 }} value={review} onChange={e => { setReview(e.target.value); setOffset(0) }}>
              <option value="">any review</option>
              <option value="pending">pending review</option>
              <option value="confirmed_ai">confirmed AI</option>
              <option value="confirmed_real">confirmed real</option>
              <option value="unresolved">unresolved</option>
            </select>
          </div>
          <select className="input" style={{ width: 168 }} value={sort} onChange={e => setSort(e.target.value as typeof sort)}>
            <option value="newest">newest first</option>
            <option value="oldest">oldest first</option>
            <option value="p_desc">most likely AI</option>
            <option value="p_asc">most likely real</option>
          </select>
        </div>

        <div className="panel-b">
          {loading && !items.length ? (
            <div className="row muted"><span className="spinner" /> loading…</div>
          ) : items.length ? (
            <div className="cards">
              {items.map(a => (
                <CaseCard
                  key={a.id}
                  a={a}
                  onOpen={() => openCase(a.id)}
                  selected={compare.includes(a.id)}
                  onSelect={() => toggleCompare(a.id)}
                />
              ))}
            </div>
          ) : (
            <div className="empty">
              <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M4 7a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
              </svg>
              No case files match. Scan an image to create one.
            </div>
          )}

          <div className="pager">
            <span>{total} cases · page {page}/{pages}</span>
            <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(o => Math.max(0, o - PAGE))}>prev</button>
            <button className="btn sm" disabled={page >= pages} onClick={() => setOffset(o => o + PAGE)}>next</button>
          </div>
        </div>
      </div>
    </div>
  )
}
