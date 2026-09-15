import { useEffect, useState } from 'react'
import { ApiError, api, type Analysis, type Health } from '../api'
import Dropzone from '../components/Dropzone'
import ResultPanel from '../components/ResultPanel'

interface Props { openId: string | null; onClearOpen: () => void; notify: (k: 'ok' | 'err', t: string) => void; compare: string[]; toggleCompare: (id: string) => void; health: Health | null }

export default function ScanView({ openId, onClearOpen, notify, compare, toggleCompare, health }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [caption, setCaption] = useState('')
  const [heatmap, setHeatmap] = useState(true)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Analysis | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!openId) return
    setBusy(true); setResult(null); setFile(null); setPreview(null)
    api.get(openId).then(a => setResult(a)).catch(e => notify('err', e.message)).finally(() => setBusy(false))
  }, [openId, notify])

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const pick = (files: File[]) => {
    const f = files[0]; setFile(f); setResult(null); setError(null); onClearOpen()
    if (preview) URL.revokeObjectURL(preview)
    setPreview(URL.createObjectURL(f))
  }

  const run = async () => {
    if (!file) return
    setBusy(true); setError(null)
    try {
      const a = await api.analyze(file, caption.trim() || undefined, heatmap)
      setResult(a)
    } catch (e) {
      const msg = e instanceof ApiError ? `${e.message}` : 'Analysis failed — is the API running?'
      setError(msg); notify('err', msg)
    } finally { setBusy(false) }
  }

  const reset = () => {
    setFile(null); setResult(null); setError(null); setCaption(''); onClearOpen()
    if (preview) URL.revokeObjectURL(preview); setPreview(null)
  }

  return (
    <div className="stack fade-in">
      {/* ── Page header ─────────────────────────────────────────── */}
      <div className="page-head entry-1">
        <div>
          <div className="eyebrow">single image · modules A B C D E</div>
          <h1>Scan an image</h1>
          <p>
            Drop a photo, product shot, artwork or screenshot. SignalScope returns a calibrated
            likelihood of AI generation, a Grad-CAM evidence map, forensic cues, provenance
            metadata and a plain-language explanation.
          </p>
        </div>
        {(result || file) && (
          <button className="btn ghost" onClick={reset}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" width="14" height="14">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" strokeLinecap="round"/>
              <path d="M3 3v5h5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            New scan
          </button>
        )}
      </div>

      {/* ── Pre-result: dropzone + options ───────────────────────── */}
      {!result && (
        <div className="grid-2 entry-2" style={{ gridTemplateColumns: '1.5fr 1fr', alignItems: 'start' }}>
          {/* Left: image zone */}
          <div className="stack">
            {!file ? (
              <Dropzone
                onFiles={pick}
                title="Drop an image to examine"
                hint="JPEG · PNG · WebP · BMP · TIFF · up to 20 MB · or paste from clipboard"
                disabled={busy}
              />
            ) : (
              <div className="frame" style={{ minHeight: 340 }}>
                {preview && <img src={preview} alt="selected image preview" />}
                {busy && <div className="scanning" aria-hidden />}
                <span className="reticle tl" /><span className="reticle tr" />
                <span className="reticle bl" /><span className="reticle br" />
              </div>
            )}
            {file && (
              <div className="row small muted mono" style={{ gap: 8, flexWrap: 'nowrap', overflow: 'hidden' }}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</span>
                <span>·</span>
                <span style={{ whiteSpace: 'nowrap' }}>{(file.size / 1024).toFixed(0)} KB</span>
                <span>·</span>
                <span style={{ whiteSpace: 'nowrap' }}>{file.type || 'unknown'}</span>
              </div>
            )}
          </div>

          {/* Right: analysis options */}
          <div className="panel" style={{ position: 'sticky', top: 24 }}>
            <div className="panel-h">
              <h3>Analysis options</h3>
              <span className={`chip ${health?.model.loaded ? 'real' : 'ai'}`}>
                {health?.model.loaded ? `GPU ${health.model.device}` : 'model offline'}
              </span>
            </div>
            <div className="panel-b stack">
              <label className="stack" style={{ gap: 6 }}>
                <span className="eyebrow">optional caption / claim <span style={{ opacity: 0.6 }}>module E</span></span>
                <textarea
                  className="input" rows={3}
                  placeholder={`e.g. "Handmade ceramic mug, brand new" — generic descriptions only, no claims about people or events`}
                  value={caption}
                  onChange={e => setCaption(e.target.value)}
                  maxLength={300}
                />
              </label>
              <label className="row" style={{ gap: 8, cursor: 'pointer' }}>
                <input type="checkbox" checked={heatmap} onChange={e => setHeatmap(e.target.checked)} />
                Produce Grad-CAM evidence heat-map
              </label>
              <div className="divider" />
              <button
                className="btn primary"
                disabled={!file || busy || !health?.model.loaded}
                onClick={run}
                style={{ justifyContent: 'center', padding: '13px 16px', fontSize: 14 }}
              >
                {busy
                  ? <><span className="spinner" /> Examining…</>
                  : <>
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="15" height="15">
                        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35" strokeLinecap="round"/>
                      </svg>
                      Run forensic analysis
                    </>
                }
              </button>
              {error && <div className="notice err">{error}</div>}
              <div className="small muted" style={{ lineHeight: 1.6 }}>
                Typical latency under a second on GPU. The verdict is a likelihood assessment;
                it is not proof and is never phrased as an accusation.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Loading a case ───────────────────────────────────────── */}
      {busy && !file && (
        <div className="row muted entry-2">
          <span className="spinner" /> Loading case…
        </div>
      )}

      {/* ── Result ──────────────────────────────────────────────── */}
      {result && (
        <ResultPanel
          analysis={result}
          onUpdate={setResult}
          notify={notify}
          inCompare={compare.includes(result.id)}
          toggleCompare={toggleCompare}
          onDeleted={reset}
        />
      )}
    </div>
  )
}
