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
      const msg = e instanceof ApiError ? `${e.message}` : 'Analysis failed - is the API running?'
      setError(msg); notify('err', msg)
    } finally { setBusy(false) }
  }

  const reset = () => { setFile(null); setResult(null); setError(null); setCaption(''); onClearOpen(); if (preview) URL.revokeObjectURL(preview); setPreview(null) }

  return (
    <div className="stack fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">single image · core task + modules A B C D E</div>
          <h1>Scan an image</h1>
          <p>Drop a photo, product shot, artwork or screenshot. SignalScope returns a calibrated likelihood that it is AI-generated, a Grad-CAM evidence map, measurable forensic cues, provenance metadata and a plain-language explanation.</p>
        </div>
        {(result || file) && <button className="btn ghost" onClick={reset}>New scan</button>}
      </div>

      {!result && (
        <div className="grid-2" style={{ gridTemplateColumns: '1.4fr 1fr' }}>
          <div className="stack">
            {!file ? (
              <Dropzone onFiles={pick} title="Drop an image to examine" hint="JPEG, PNG, WebP, BMP, TIFF · up to 20 MB · or paste from clipboard" disabled={busy} />
            ) : (
              <div className="frame" style={{ minHeight: 320 }}>
                {preview && <img src={preview} alt="selected image preview" />}
                {busy && <div className="scanning" aria-hidden />}
                <span className="reticle tl" /><span className="reticle tr" /><span className="reticle bl" /><span className="reticle br" />
              </div>
            )}
            {file && <div className="row small muted mono"><span>{file.name}</span><span>·</span><span>{(file.size / 1024).toFixed(0)} KB</span><span>·</span><span>{file.type || 'unknown type'}</span></div>}
          </div>
          <div className="panel">
            <div className="panel-h"><h3>Analysis options</h3><span className="chip">{health?.model.loaded ? `GPU ${health.model.device}` : 'model offline'}</span></div>
            <div className="panel-b stack">
              <label className="stack" style={{ gap: 6 }}>
                <span className="eyebrow">optional caption / claim (module E)</span>
                <textarea className="input" rows={3} placeholder='e.g. "Handmade ceramic mug, brand new" - generic descriptions only, no claims about people or events' value={caption} onChange={e => setCaption(e.target.value)} maxLength={300} />
              </label>
              <label className="row" style={{ gap: 8 }}><input type="checkbox" checked={heatmap} onChange={e => setHeatmap(e.target.checked)} /> Produce Grad-CAM evidence heat-map</label>
              <div className="divider" />
              <button className="btn primary" disabled={!file || busy || !health?.model.loaded} onClick={run} style={{ justifyContent: 'center', padding: '12px 16px' }}>
                {busy ? <><span className="spinner" /> Examining…</> : 'Run forensic analysis'}
              </button>
              {error && <div className="notice err">{error}</div>}
              <div className="small muted">Typical latency under a second on GPU. The verdict is a likelihood assessment produced by a statistical model; it is not proof and is never phrased as an accusation.</div>
            </div>
          </div>
        </div>
      )}

      {busy && !file && <div className="row muted"><span className="spinner" /> Loading case…</div>}
      {result && <ResultPanel analysis={result} onUpdate={setResult} notify={notify} inCompare={compare.includes(result.id)} toggleCompare={toggleCompare} onDeleted={reset} />}
    </div>
  )
}
