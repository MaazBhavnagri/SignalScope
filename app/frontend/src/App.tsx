import { useCallback, useEffect, useState } from 'react'
import { api, type Health } from './api'
import ScanView from './views/ScanView'
import BatchView from './views/BatchView'
import CasesView from './views/CasesView'
import ModelCardView from './views/ModelCardView'

export type View = 'scan' | 'batch' | 'cases' | 'model'
export interface Toast { kind: 'ok' | 'err'; text: string }

const NAV: { id: View; label: string; key: string; icon: React.ReactNode }[] = [
  {
    id: 'scan', label: 'Scan', key: '1',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <circle cx="12" cy="12" r="7" />
        <path d="M12 2v3M12 19v3M2 12h3M19 12h3" strokeLinecap="round" />
        <circle cx="12" cy="12" r="2.5" fill="currentColor" stroke="none" />
      </svg>
    ),
  },
  {
    id: 'batch', label: 'Batch', key: '2',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <rect x="3" y="3" width="7.5" height="7.5" rx="1.5" />
        <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" />
        <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" />
        <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" />
      </svg>
    ),
  },
  {
    id: 'cases', label: 'Cases', key: '3',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M4 7a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
        <path d="M4 11h16" />
      </svg>
    ),
  },
  {
    id: 'model', label: 'Model', key: '4',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M4 19V5M4 19h16" strokeLinecap="round" />
        <path d="M7 15l4-5 3 3 5-7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
]

export default function App() {
  const [view, setView] = useState<View>('scan')
  const [health, setHealth] = useState<Health | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const [compare, setCompare] = useState<string[]>([])
  const [toast, setToast] = useState<Toast | null>(null)

  const notify = useCallback((kind: Toast['kind'], text: string) => {
    setToast({ kind, text })
    window.setTimeout(() => setToast(null), 4200)
  }, [])

  useEffect(() => {
    let alive = true
    const tick = () => api.health().then(h => alive && setHealth(h)).catch(() => alive && setHealth(null))
    tick(); const id = window.setInterval(tick, 15000)
    return () => { alive = false; window.clearInterval(id) }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName?.match(/INPUT|TEXTAREA|SELECT/)) return
      const n = NAV.find(n => n.key === e.key); if (n) setView(n.id)
    }
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey)
  }, [])

  const openCase = useCallback((id: string) => { setOpenId(id); setView('scan') }, [])
  const toggleCompare = useCallback((id: string) => {
    setCompare(c => c.includes(id) ? c.filter(x => x !== id) : [...c.slice(-1), id])
  }, [])

  const ok = health?.model.loaded
  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 64 64">
            <rect width="64" height="64" rx="14" fill="#0e1522" stroke="rgba(255,255,255,0.10)" />
            <circle cx="32" cy="32" r="20" fill="none" stroke="var(--amber)" strokeWidth="1.5" opacity="0.85" />
            <circle cx="32" cy="32" r="12" fill="none" stroke="rgba(245,158,11,0.30)" strokeDasharray="2 3.5" />
            <path d="M32 4v9M32 51v9M4 32h9M51 32h9" stroke="var(--amber)" strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="32" cy="32" r="4" fill="var(--amber-2)" style={{ filter: 'drop-shadow(0 0 8px var(--amber))' }} />
          </svg>
          <div>
            <div className="brand-name">SignalScope</div>
            <div className="brand-sub">media forensics</div>
          </div>
        </div>

        <nav className="nav" aria-label="Primary">
          {NAV.map(n => (
            <button
              key={n.id}
              className={view === n.id ? 'active' : ''}
              onClick={() => { setView(n.id); if (n.id === 'scan') setOpenId(null) }}
            >
              {n.icon}
              <span>{n.label}</span>
              <span className="k">{n.key}</span>
            </button>
          ))}
        </nav>

        <div className="rail-foot">
          <div className="status-line">
            <span className={`dot ${health ? (ok ? 'ok' : 'bad') : ''}`} />
            {health
              ? (ok ? `model live · ${health.model.device}` : 'model unavailable')
              : 'api offline'}
          </div>
          {ok && (
            <div className="status-line">
              thr {health!.model.threshold?.toFixed(2)} · T {health!.model.temperature?.toFixed(2)} · val AUC {health!.model.val_auc?.toFixed(3)}
            </div>
          )}
          {health && !ok && <div className="small" style={{ color: 'var(--ai)' }}>{health.model.error}</div>}
          <div className="small muted" style={{ lineHeight: 1.55 }}>
            Outputs are likelihoods, not accusations. No claims about real people or events.
          </div>
        </div>
      </aside>

      <main className="main">
        {view === 'scan'  && <ScanView openId={openId} onClearOpen={() => setOpenId(null)} notify={notify} compare={compare} toggleCompare={toggleCompare} health={health} />}
        {view === 'batch' && <BatchView notify={notify} openCase={openCase} />}
        {view === 'cases' && <CasesView notify={notify} openCase={openCase} compare={compare} toggleCompare={toggleCompare} setCompare={setCompare} />}
        {view === 'model' && <ModelCardView />}
      </main>

      {toast && <div className={`toast ${toast.kind}`} role="status">{toast.text}</div>}
    </div>
  )
}
