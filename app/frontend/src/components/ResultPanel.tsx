import { Fragment, useState } from 'react'
import { api, fmtBytes, fmtDate, pct, tone, type Analysis, type Cue } from '../api'
import Gauge from './Gauge'

interface Props { analysis: Analysis; onUpdate: (a: Analysis) => void; notify: (k: 'ok' | 'err', t: string) => void; inCompare: boolean; toggleCompare: (id: string) => void; onDeleted?: () => void }

const SIGNAL_LABEL: Record<string, string> = {
  supports_ai: 'declares AI generation',
  supports_real: 'camera capture data',
  neutral: 'no decisive metadata',
  stripped: 'metadata stripped',
}
const STAB_TONE: Record<string, string> = { stable: 'real', moderate: 'unsure', fragile: 'ai' }

function CueRange({ c }: { c: Cue }) {
  const [lo, hi] = c.typical_range
  const span = Math.max(hi - lo, 1e-9)
  const norm = (v: number) => Math.min(1, Math.max(0, (v - (lo - span)) / (3 * span)))
  return (
    <div className="range" title={`typical ${lo} – ${hi}`}>
      <div className="band" style={{ left: `${norm(lo) * 100}%`, width: `${(norm(hi) - norm(lo)) * 100}%` }} />
      <div className={`mark ${c.fired ? 'fired' : ''}`} style={{ left: `calc(${norm(c.value) * 100}% - 1px)` }} />
    </div>
  )
}

export default function ResultPanel({ analysis: a, onUpdate, notify, inCompare, toggleCompare, onDeleted }: Props) {
  const [opacity, setOpacity] = useState(0.75)
  const [showHeat, setShowHeat] = useState(true)
  const [showHot, setShowHot] = useState(true)
  const [showGrid, setShowGrid] = useState(false)
  const [stressing, setStressing] = useState(false)
  const [note, setNote] = useState(a.review_note ?? '')
  const [saving, setSaving] = useState(false)

  const t = tone(a.band)
  const isAI = a.verdict === 'ai_generated'
  const attrApplicable = a.attribution && a.attribution.applicable !== false
  const fired = a.cues.filter(c => c.fired)

  const stress = async () => {
    setStressing(true)
    try {
      const r = await api.robustness(a.id)
      onUpdate({ ...a, robustness: r, has_robustness: true, stability: r.stability })
      notify('ok', 'Stress test complete')
    } catch (e) { notify('err', (e as Error).message) } finally { setStressing(false) }
  }

  const review = async (decision: string | null) => {
    setSaving(true)
    try {
      const r = await api.review(a.id, decision, note.trim() || null)
      onUpdate(r)
      notify('ok', decision ? 'Reviewer decision saved' : 'Review cleared')
    } catch (e) { notify('err', (e as Error).message) } finally { setSaving(false) }
  }

  const remove = async () => {
    if (!window.confirm('Delete this case file and its stored image?')) return
    try { await api.remove(a.id); notify('ok', 'Case deleted'); onDeleted?.() }
    catch (e) { notify('err', (e as Error).message) }
  }

  return (
    <div className="stack fade-in">
      {/* ── Case header bar ────────────────────────────────────── */}
      <div className="row entry-1" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="row mono small muted" style={{ gap: 6, flexWrap: 'nowrap', overflow: 'hidden' }}>
          <span style={{ color: 'var(--amber-2)', fontWeight: 600 }}>CASE {a.id.slice(0, 8)}</span>
          <span>·</span>
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.filename}</span>
          <span>·</span>
          <span style={{ whiteSpace: 'nowrap' }}>{a.width}×{a.height}</span>
          <span>·</span>
          <span style={{ whiteSpace: 'nowrap' }}>{fmtBytes(a.size_bytes)}</span>
          <span>·</span>
          <span style={{ whiteSpace: 'nowrap' }}>{fmtDate(a.created_at)}</span>
          {a.latency_ms != null && <><span>·</span><span style={{ whiteSpace: 'nowrap' }}>{a.latency_ms.toFixed(0)} ms</span></>}
        </div>
        <div className="row" style={{ flexShrink: 0, gap: 6 }}>
          <button className={`btn sm ${inCompare ? 'primary' : ''}`} onClick={() => toggleCompare(a.id)}>
            {inCompare ? 'In compare tray' : 'Compare'}
          </button>
          <a className="btn sm" href={api.reportUrl(a.id, 'html')} target="_blank" rel="noreferrer">Report</a>
          <a className="btn sm" href={api.reportUrl(a.id, 'json')} download>JSON</a>
          <button className="btn sm danger" onClick={remove}>Delete</button>
        </div>
      </div>

      {/* ── Verdict hero ───────────────────────────────────────── */}
      <div className="paper entry-2" style={{ padding: '28px 32px' }}>
        <div className="verdict">
          <Gauge p={a.probability_ai} threshold={a.threshold} band={a.band} light />
          <div>
            <span className={`stamp ${t}`}>{a.label}</span>
            <div className="verdict-title" style={{ marginTop: 12 }}>{a.explanation.summary}</div>
            <div className="verdict-sub">
              {a.band === 'uncertain'
                ? 'The image sits close to the decision threshold. Do not act on this verdict alone.'
                : isAI
                  ? 'A likelihood assessment — not proof and not an accusation. Verify with provenance and context before acting.'
                  : 'No strong synthetic signature was detected. Absence of evidence is not proof of authenticity.'}
            </div>
            <div className="readout">
              <span>calibrated <b>{pct(a.probability_ai, 1)}</b></span>
              <span>raw <b>{pct(a.probability_ai_uncalibrated, 1)}</b></span>
              <span>threshold <b>{a.threshold.toFixed(2)}</b></span>
              <span>band <b>{a.band.replace('_', ' ')}</b></span>
              {a.combined && a.combined.combined_probability !== a.probability_ai && (
                <span>with metadata <b>{pct(a.combined.combined_probability, 1)}</b></span>
              )}
              {a.review_decision && <span>reviewer <b>{a.review_decision.replace('_', ' ')}</b></span>}
            </div>
          </div>
        </div>
      </div>

      {/* ── Evidence map + explanation ─────────────────────────── */}
      <div className="grid-2 entry-3" style={{ gridTemplateColumns: '1.35fr 1fr', alignItems: 'start' }}>
        {/* Evidence map */}
        <div className="panel">
          <div className="panel-h">
            <h3>
              Evidence map
              <span className="eyebrow">grad-cam · ai logit</span>
            </h3>
            {a.regions && (
              <span className={`chip ${a.regions.concentration === 'concentrated' ? 'amber' : ''}`}>
                {a.regions.concentration} · {a.regions.top_regions.join(' & ')}
              </span>
            )}
          </div>
          <div className="panel-b">
            <div className="frame">
              <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
                <img src={a.image_url} alt={a.filename} style={{ minWidth: 'min(100%, 420px)' }} />
                {a.heatmap_url && showHeat && (
                  <img className="overlay" src={a.heatmap_url} alt="" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity }} />
                )}
                {showHot && a.regions?.hot_bbox && (
                  <div
                    className="hot" data-label="peak evidence"
                    style={{
                      left: `${a.regions.hot_bbox[0] * 100}%`,
                      top: `${a.regions.hot_bbox[1] * 100}%`,
                      width: `${(a.regions.hot_bbox[2] - a.regions.hot_bbox[0]) * 100}%`,
                      height: `${(a.regions.hot_bbox[3] - a.regions.hot_bbox[1]) * 100}%`,
                    }}
                  />
                )}
                {showGrid && a.regions && (
                  <div className="grid9" style={{ inset: 0 }}>
                    {[...a.regions.cells]
                      .sort((x, y) => x.row * 3 + x.col - (y.row * 3 + y.col))
                      .map(c => <div key={c.region}>{Math.round(c.share * 100)}%</div>)}
                  </div>
                )}
              </div>
              <span className="reticle tl" /><span className="reticle tr" />
              <span className="reticle bl" /><span className="reticle br" />
            </div>
            <div className="frame-tools">
              <label><input type="checkbox" checked={showHeat} onChange={e => setShowHeat(e.target.checked)} disabled={!a.heatmap_url} /> heat-map</label>
              <label>opacity <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={e => setOpacity(Number(e.target.value))} disabled={!showHeat} /></label>
              <label><input type="checkbox" checked={showHot} onChange={e => setShowHot(e.target.checked)} /> peak region</label>
              <label><input type="checkbox" checked={showGrid} onChange={e => setShowGrid(e.target.checked)} /> 3×3 grid</label>
            </div>
            <div className="small muted" style={{ marginTop: 10, lineHeight: 1.6 }}>
              {isAI || a.band === 'uncertain'
                ? 'Warm areas contributed most to the "AI-generated" score. '
                : 'Low, scattered activation: the model found no region with a strong synthetic signature. '}
              {a.regions && `Top two regions hold ${pct(a.regions.top2_share)} of the heat-map mass; hot area covers ${pct(a.regions.hot_area_fraction)} of the frame.`}
            </div>
          </div>
        </div>

        {/* Explanation + robustness */}
        <div className="stack">
          <div className="paper" style={{ padding: '18px 22px' }}>
            <div className="eyebrow" style={{ marginBottom: 12 }}>explanation · grounded in measured cues</div>
            <ol className="evidence-list">
              {a.explanation.sentences.map((s, i) => (
                <li key={i}><span className="n">{String(i + 1).padStart(2, '0')}</span><span>{s}</span></li>
              ))}
            </ol>
          </div>

          {/* Robustness */}
          <div className="panel">
            <div className="panel-h">
              <h3>Robustness stress test <span className="eyebrow">module c</span></h3>
              {a.robustness
                ? <span className={`chip ${STAB_TONE[a.robustness.stability]}`}>{a.robustness.stability}</span>
                : <button className="btn sm" onClick={stress} disabled={stressing}>{stressing ? <><span className="spinner" /> testing</> : 'Run stress test'}</button>
              }
            </div>
            {a.robustness && (
              <div className="panel-b stack" style={{ gap: 10 }}>
                <div className="small" style={{ lineHeight: 1.6 }}>{a.robustness.note}</div>
                <table className="table">
                  <thead><tr><th>Degradation</th><th className="num">P(AI)</th><th className="num">Δ</th><th>Verdict</th></tr></thead>
                  <tbody>{a.robustness.items.map(i => (
                    <tr key={i.id} className={i.flipped ? 'hl' : ''}>
                      <td>{i.label}</td>
                      <td className="num">{i.probability_ai.toFixed(3)}</td>
                      <td className="num" style={{ color: Math.abs(i.delta) > 0.1 ? 'var(--unsure)' : undefined }}>
                        {i.delta >= 0 ? '+' : ''}{i.delta.toFixed(3)}
                      </td>
                      <td>
                        <span className={`chip ${i.verdict === 'ai_generated' ? 'ai' : 'real'}`}>
                          {i.verdict === 'ai_generated' ? 'AI' : 'real'}{i.flipped ? ' · flipped' : ''}
                        </span>
                      </td>
                    </tr>
                  ))}</tbody>
                </table>
                <button className="btn sm ghost" onClick={stress} disabled={stressing}>re-run</button>
              </div>
            )}
            {!a.robustness && (
              <div className="panel-b small muted" style={{ lineHeight: 1.65 }}>
                Re-analyses this image after JPEG re-compression, downscaling, blur, noise and a simulated
                screenshot to show whether the verdict survives real-world degradation.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Forensic cue ledger ────────────────────────────────── */}
      <div className="panel entry-4">
        <div className="panel-h">
          <h3>Forensic cue ledger <span className="eyebrow">measured statistics vs real-photo reference</span></h3>
          <span className={`chip ${fired.length > 0 ? 'ai' : 'real'}`}>{fired.length} of {a.cues.length} outside range</span>
        </div>
        <div className="panel-b" style={{ overflowX: 'auto' }}>
          <table className="ledger">
            <thead>
              <tr>
                <th>Cue</th><th>Measured</th><th>Real-photo range (5–95th pct)</th><th>Finding</th>
              </tr>
            </thead>
            <tbody>
              {a.cues.map(c => (
                <tr key={c.id} className={c.fired ? 'fired' : ''}>
                  <td style={{ whiteSpace: 'nowrap' }}>{c.name}</td>
                  <td className="mono">{c.value}</td>
                  <td>
                    <div className="row" style={{ gap: 10, flexWrap: 'nowrap' }}>
                      <CueRange c={c} />
                      <span className="mono small muted" style={{ whiteSpace: 'nowrap' }}>{c.typical_range[0]} – {c.typical_range[1]}</span>
                    </div>
                  </td>
                  <td style={{ color: c.fired ? 'var(--ink)' : 'var(--ink-3)' }}>
                    {c.finding}
                    {c.fired && <span className="chip ai" style={{ marginLeft: 8 }}>{c.side} · {Math.round(c.strength * 100)}%</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 10, lineHeight: 1.6 }}>
            Reference ranges are the 5th–95th percentiles measured on real validation photographs.
            A cue outside the range is a verifiable, reproducible observation — not a verdict on its own.
          </div>
        </div>
      </div>

      {/* ── Attribution + Provenance + Caption + Reviewer ─────── */}
      <div className="grid-3 entry-5" style={{ alignItems: 'start' }}>
        {/* Attribution */}
        <div className="panel">
          <div className="panel-h">
            <h3>Generator attribution <span className="eyebrow">module b</span></h3>
            {attrApplicable && a.attribution && <span className="chip amber">{a.attribution.family}</span>}
          </div>
          <div className="panel-b stack" style={{ gap: 12 }}>
            {attrApplicable && a.attribution ? (
              <>
                <div className="small" style={{ lineHeight: 1.65 }}>
                  Artefact pattern most consistent with <b>{a.attribution.family}</b> (closest known generator{' '}
                  {a.attribution.generator}, {pct(a.attribution.confidence)} of attribution mass).
                </div>
                <div className="bars">
                  {Object.entries(a.attribution.family_distribution).map(([k, v]) => (
                    <div className="bar" key={k}>
                      <span className="n">{k}</span>
                      <div className="track"><div className="fill" style={{ width: `${v * 100}%` }} /></div>
                      <span className="v">{pct(v)}</span>
                    </div>
                  ))}
                </div>
                <details>
                  <summary className="small muted" style={{ cursor: 'pointer' }}>per-generator distribution</summary>
                  <div className="bars" style={{ marginTop: 8 }}>
                    {Object.entries(a.attribution.distribution).map(([k, v]) => (
                      <div className="bar" key={k}>
                        <span className="n">{k}</span>
                        <div className="track"><div className="fill" style={{ width: `${v * 100}%` }} /></div>
                        <span className="v">{pct(v)}</span>
                      </div>
                    ))}
                  </div>
                </details>
                <div className="small muted">{a.attribution.note}</div>
              </>
            ) : (
              <div className="small muted" style={{ lineHeight: 1.65 }}>
                Attribution is only reported when the verdict leans AI-generated. For a likely-real image the
                generator head has no meaningful answer.
              </div>
            )}
          </div>
        </div>

        {/* Provenance */}
        <div className="panel">
          <div className="panel-h">
            <h3>Provenance &amp; metadata <span className="eyebrow">module d</span></h3>
            {a.provenance && (
              <span className={`chip ${a.provenance.signal === 'supports_ai' ? 'ai' : a.provenance.signal === 'supports_real' ? 'real' : ''}`}>
                {SIGNAL_LABEL[a.provenance.signal] ?? a.provenance.signal}
              </span>
            )}
          </div>
          <div className="panel-b stack" style={{ gap: 12 }}>
            {a.provenance ? (
              <>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.7 }}>
                  {a.provenance.reasons.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
                <dl className="kv">
                  <dt>format</dt>
                  <dd className="mono">
                    {a.provenance.format} · {a.provenance.mode}
                    {a.provenance.jpeg_quality_estimate ? ` · JPEG q≈${a.provenance.jpeg_quality_estimate}` : ''}
                  </dd>
                  <dt>exif fields</dt>
                  <dd className="mono">{a.provenance.exif_field_count}{a.provenance.gps_present ? ' · GPS present (not read)' : ''}</dd>
                  {Object.entries(a.provenance.camera).slice(0, 6).map(([k, v]) => (
                    <Fragment key={k}><dt>{k}</dt><dd className="mono">{String(Array.isArray(v) ? v.join('/') : v)}</dd></Fragment>
                  ))}
                  <dt>c2pa</dt>
                  <dd>{a.provenance.c2pa.present
                    ? `manifest present${a.provenance.c2pa.claim_generator ? ` · ${a.provenance.c2pa.claim_generator}` : ''} (signature not validated)`
                    : 'no Content Credentials manifest'}
                  </dd>
                </dl>
                {a.combined && (
                  <div className="notice" style={{ fontSize: 12, lineHeight: 1.55 }}>
                    <b>Fusion rule:</b> {a.combined.rule}
                    {a.combined.conflict ? ' · metadata and visual evidence disagree — metadata can be copied or stripped, so the visual verdict is shown as primary.' : ''}
                  </div>
                )}
              </>
            ) : (
              <div className="small muted">No file bytes available for metadata analysis.</div>
            )}
          </div>
        </div>

        {/* Caption + Reviewer */}
        <div className="stack">
          <div className="panel">
            <div className="panel-h">
              <h3>Caption consistency <span className="eyebrow">module e</span></h3>
              {a.caption_consistency?.band && (
                <span className={`chip ${a.caption_consistency.band === 'consistent' ? 'real' : a.caption_consistency.band === 'inconsistent' ? 'ai' : 'unsure'}`}>
                  {a.caption_consistency.band}
                </span>
              )}
            </div>
            <div className="panel-b small">
              {a.caption_consistency?.band ? (
                <div className="stack" style={{ gap: 8 }}>
                  <div style={{ lineHeight: 1.6 }}>"{a.caption_consistency.caption}"</div>
                  <div className="row mono muted">
                    <span>similarity {a.caption_consistency.similarity?.toFixed(3)}</span>
                    <span>·</span>
                    <span>rank {a.caption_consistency.rank_vs_distractors}/{(a.caption_consistency.n_distractors ?? 0) + 1}</span>
                  </div>
                  <div style={{ lineHeight: 1.6 }}>{a.caption_consistency.sentence}</div>
                </div>
              ) : a.caption_consistency?.error ? (
                <div className="muted">{a.caption_consistency.error}</div>
              ) : (
                <div className="muted" style={{ lineHeight: 1.65 }}>
                  No caption supplied. Add a generic description at scan time to check whether the text matches the image.
                </div>
              )}
            </div>
          </div>

          <div className="panel">
            <div className="panel-h">
              <h3>Reviewer decision <span className="eyebrow">human in the loop</span></h3>
              {a.review_decision && <span className="chip amber">{a.review_decision.replace('_', ' ')}</span>}
            </div>
            <div className="panel-b stack" style={{ gap: 10 }}>
              <textarea
                className="input" rows={2}
                placeholder="Reviewer note (context, source, decision rationale)"
                value={note}
                onChange={e => setNote(e.target.value)}
              />
              <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
                <button className="btn sm" disabled={saving} onClick={() => review('confirmed_ai')}>Confirm AI-generated</button>
                <button className="btn sm" disabled={saving} onClick={() => review('confirmed_real')}>Confirm real</button>
                <button className="btn sm ghost" disabled={saving} onClick={() => review('unresolved')}>Unresolved</button>
                {a.review_decision && <button className="btn sm ghost" disabled={saving} onClick={() => review(null)}>Clear</button>}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
