import { Fragment, useEffect, useState } from 'react'
import { api } from '../api'

type Any = Record<string, any>

function RocChart({ groups }: { groups: Any }) {
  const W = 320, H = 280, P = 34
  const x = (v: number) => P + v * (W - P - 10), y = (v: number) => H - P - v * (H - P - 10)
  const series: { key: string; color: string; label: string }[] = [
    { key: 'unseen', color: 'var(--ai-2)', label: 'unseen generators' }, { key: 'seen', color: 'var(--amber-2)', label: 'seen generators' }, { key: 'cifake', color: 'var(--real-2)', label: 'CIFAKE test' },
  ].filter(s => groups[s.key])
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 420 }} role="img" aria-label="ROC curves">
      <rect x={P} y={10} width={W - P - 10} height={H - P - 10} fill="rgba(255,255,255,0.01)" stroke="var(--line)" rx="4" />
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--line-2)" strokeDasharray="4 4" />
      {series.map(s => {
        const r = groups[s.key].roc as { fpr: number[]; tpr: number[] }
        const d = r.fpr.map((f, i) => `${i ? 'L' : 'M'} ${x(f)} ${y(r.tpr[i])}`).join(' ')
        return <path key={s.key} d={d} fill="none" stroke={s.color} strokeWidth={2.2} style={{ filter: 'drop-shadow(0 0 6px currentColor)' }} />
      })}
      {series.map((s, i) => <g key={s.key}><rect x={x(0.42)} y={y(0.28) + i * 16 - 8} width={10} height={3} rx={1} fill={s.color} /><text x={x(0.42) + 15} y={y(0.28) + i * 16 - 4} fontSize={10} fill="var(--ink-2)" fontFamily="var(--mono)">{s.label} · AUC {groups[s.key]?.auc?.toFixed(3) ?? '-'}</text></g>)}
      <text x={x(0.5)} y={H - 8} fontSize={10} textAnchor="middle" fill="var(--ink-3)" fontFamily="var(--mono)">false-positive rate (real flagged as AI)</text>
      <text x={10} y={y(0.5)} fontSize={10} textAnchor="middle" fill="var(--ink-3)" fontFamily="var(--mono)" transform={`rotate(-90 10 ${y(0.5)})`}>true-positive rate</text>
    </svg>
  )
}

function ConfMatrix({ cm, title }: { cm: Any; title: string }) {
  const cells = [[cm.tn, cm.fp], [cm.fn, cm.tp]]
  const max = Math.max(cm.tn, cm.fp, cm.fn, cm.tp, 1)
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 8 }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 1fr', gap: 6, fontSize: 12 }}>
        <div /><div className="mono small muted" style={{ textAlign: 'center' }}>pred real</div><div className="mono small muted" style={{ textAlign: 'center' }}>pred AI</div>
        {cells.map((row, i) => (<Fragment key={i}>
          <div className="mono small muted" style={{ alignSelf: 'center' }}>{i ? 'true AI' : 'true real'}</div>
          {row.map((v, j) => (
            <div key={j} className="mono" style={{
              padding: '14px 8px',
              textAlign: 'center',
              borderRadius: 6,
              background: i === j ? `rgba(16, 185, 129, ${0.12 + 0.5 * v / max})` : `rgba(244, 63, 94, ${0.12 + 0.5 * v / max})`,
              border: i === j ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(244, 63, 94, 0.3)',
              color: '#fff',
              fontWeight: 600,
            }}>
              {v.toLocaleString()}
            </div>
          ))}
        </Fragment>))}
      </div>
    </div>
  )
}

export default function ModelCardView() {
  const [card, setCard] = useState<Any | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { api.modelCard().then(setCard).catch(e => setErr(e.message)) }, [])
  if (err) return <div className="notice err">{err}</div>
  if (!card) return <div className="row muted"><span className="spinner" /> loading model card…</div>
  const m = card.metrics as Any | null
  const groups: Any = m?.groups ?? {}
  const rob: Any = card.robustness?.results ?? null
  const atk: Any = card.attacks ?? null
  const calib: Any = card.calibration ?? null
  const hist: Any[] = card.train_history ?? []
  const ds: Any = card.dataset_card ?? null
  const gens: [string, Any][] = Object.entries(m?.per_generator ?? {})

  return (
    <div className="stack fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">model card · honest reporting</div>
          <h1>Model card &amp; evaluation</h1>
          <p>Metrics on our local held-out test split under the unseen-generator protocol: {card.protocol.held_out_generators.join(' and ')} were never used for training or calibration. The organisers' held-out set is evaluated separately via the predict interface.</p>
        </div>
        <div className="row"><span className="chip">{card.model?.checkpoint?.split(/[\\/]/).pop()}</span><span className="chip">epoch {card.model?.epoch ?? '?'}</span><span className="chip amber">{card.model?.device}</span></div>
      </div>

      {!m && <div className="notice">No metrics.json yet - run <code>python -m model.evaluate</code> after training.</div>}

      {m && (<>
        <div className="strip">
          {['unseen', 'seen', 'overall', 'cifake'].filter(k => groups[k]).map(k => (
            <div className="stat" key={k} style={k === 'unseen' ? { borderColor: 'var(--amber)' } : undefined}>
              <div className="eyebrow">{k === 'unseen' ? 'AUC · unseen generators' : k === 'seen' ? 'AUC · seen generators' : k === 'overall' ? 'AUC · overall test' : 'AUC · CIFAKE test'}</div>
              <div className="v" style={k === 'unseen' ? { color: 'var(--amber-2)' } : undefined}>{groups[k].auc?.toFixed(4) ?? '-'}</div>
              <div className="small muted mono">F1 {groups[k].macro_f1?.toFixed(3) ?? '-'} · acc {groups[k].accuracy != null ? (groups[k].accuracy * 100).toFixed(1) + '%' : '-'} · FPR {groups[k].fpr != null ? (groups[k].fpr * 100).toFixed(1) + '%' : '-'}</div>
            </div>
          ))}
          {m.attribution && <div className="stat"><div className="eyebrow">attribution acc (seen)</div><div className="v">{(m.attribution.accuracy * 100).toFixed(1)}<small>%</small></div><div className="small muted mono">macro-F1 {m.attribution.macro_f1.toFixed(3)} · family {(m.attribution.family_accuracy * 100).toFixed(0)}%</div></div>}
          {m.calibration && <div className="stat"><div className="eyebrow">calibration ECE</div><div className="v">{m.calibration.ece.toFixed(3)}</div><div className="small muted mono">T {m.temperature.toFixed(2)} · thr {m.threshold.toFixed(2)}</div></div>}
        </div>

        <div className="grid-2" style={{ alignItems: 'start' }}>
          <div className="panel"><div className="panel-h"><h3>ROC curves</h3></div><div className="panel-b"><RocChart groups={groups} /></div></div>
          <div className="panel"><div className="panel-h"><h3>Operating point <span className="eyebrow">threshold {m.threshold.toFixed(2)} · target FPR {(card.protocol.target_fpr * 100).toFixed(0)}%</span></h3></div>
            <div className="panel-b stack">
              <table className="table"><thead><tr><th>split</th><th className="num">n</th><th className="num">AUC</th><th className="num">macro-F1</th><th className="num">acc</th><th className="num">FPR</th><th className="num">TPR</th></tr></thead>
                <tbody>{Object.entries(groups).map(([k, g]: [string, Any]) => <tr key={k} className={k === 'unseen' ? 'hl' : ''}><td>{k}</td><td className="num">{g.n}</td><td className="num">{g.auc?.toFixed(4) ?? '-'}</td><td className="num">{g.macro_f1?.toFixed(3) ?? '-'}</td><td className="num">{g.accuracy != null ? (g.accuracy * 100).toFixed(1) + '%' : '-'}</td><td className="num">{g.fpr != null ? (g.fpr * 100).toFixed(1) + '%' : '-'}</td><td className="num">{g.tpr != null ? (g.tpr * 100).toFixed(1) + '%' : '-'}</td></tr>)}</tbody></table>
              <div className="grid-2">{groups.unseen && <ConfMatrix cm={groups.unseen.confusion_matrix} title="confusion · unseen split" />}{groups.overall && <ConfMatrix cm={groups.overall.confusion_matrix} title="confusion · overall" />}</div>
            </div>
          </div>
        </div>

        <div className="grid-2" style={{ alignItems: 'start' }}>
          <div className="panel"><div className="panel-h"><h3>Per-generator AUC <span className="eyebrow">fakes vs high-res test reals</span></h3></div>
            <div className="panel-b bars">{gens.sort((a, b) => b[1].auc - a[1].auc).map(([g, v]) => (
              <div className="bar" key={g} style={{ gridTemplateColumns: '160px 1fr 60px' }}><span className="n">{g} {v.held_out && <span className="chip amber" style={{ padding: '0 6px', fontSize: 9 }}>unseen</span>}</span><div className="track"><div className={`fill ${v.held_out ? '' : 'real'}`} style={{ width: `${Math.max(0, (v.auc - 0.5) * 200)}%` }} /></div><span className="v">{v.auc.toFixed(3)}</span></div>))}
              <div className="small muted">Bar length = AUC above chance (0.5 → 1.0). Detection rate at threshold: {gens.map(([g, v]) => `${g} ${(v.detection_rate * 100).toFixed(0)}%`).join(' · ')}</div>
            </div>
          </div>
          <div className="panel"><div className="panel-h"><h3>Robustness to degradation <span className="eyebrow">module c</span></h3></div>
            <div className="panel-b">{rob ? (
              <table className="table"><thead><tr><th>degradation</th><th className="num">AUC seen</th><th className="num">AUC unseen</th><th className="num">acc</th><th className="num">FPR</th><th className="num">flips</th></tr></thead>
                <tbody>{Object.values(rob).map((r: Any) => <tr key={r.label}><td>{r.label}</td><td className="num">{r.auc_seen?.toFixed(3) ?? '-'}</td><td className="num" style={{ color: r.auc_unseen < 0.8 ? 'var(--ai)' : undefined }}>{r.auc_unseen?.toFixed(3) ?? '-'}</td><td className="num">{r.accuracy != null ? (r.accuracy * 100).toFixed(1) + '%' : '-'}</td><td className="num">{r.fpr != null ? (r.fpr * 100).toFixed(1) + '%' : '-'}</td><td className="num">{r.verdict_flip_rate != null ? (r.verdict_flip_rate * 100).toFixed(1) + '%' : '-'}</td></tr>)}</tbody></table>
            ) : <div className="small muted">run <code>python -m model.robustness</code></div>}</div>
          </div>
        </div>

        <div className="grid-2" style={{ alignItems: 'start' }}>
          <div className="panel"><div className="panel-h"><h3>Active-defence analysis <span className="eyebrow">module g · white-box attacks</span></h3></div>
            <div className="panel-b">{atk ? (<>
              <table className="table"><thead><tr><th>attack</th><th className="num">detection</th><th className="num">+JPEG75</th><th className="num">+TTA</th><th className="num">+both</th></tr></thead>
                <tbody>{Object.entries(atk.detection_rate as Record<string, Any>).map(([k, r]) => <tr key={k}><td>{k}</td><td className="num" style={{ color: r.plain < 0.5 ? 'var(--ai)' : undefined }}>{(r.plain * 100).toFixed(1)}%</td><td className="num">{(r.jpeg75 * 100).toFixed(1)}%</td><td className="num">{(r.tta * 100).toFixed(1)}%</td><td className="num">{(r['jpeg75+tta'] * 100).toFixed(1)}%</td></tr>)}</tbody></table>
              <div className="small muted" style={{ marginTop: 8 }}>FPR on real images with each mitigation: {Object.entries(atk.fpr_on_reals as Record<string, number>).map(([k, v]) => `${k} ${(v * 100).toFixed(1)}%`).join(' · ')}. {atk.notes?.[0]}</div>
            </>) : <div className="small muted">run <code>python -m model.attacks</code></div>}</div>
          </div>
          <div className="panel"><div className="panel-h"><h3>Calibration &amp; training</h3></div>
            <div className="panel-b stack">
              {calib && <dl className="kv"><dt>temperature</dt><dd className="mono">{calib.temperature.toFixed(3)}</dd><dt>ECE before → after</dt><dd className="mono">{calib.ece_before.toFixed(4)} → {calib.ece_after.toFixed(4)}</dd><dt>val FPR @ thr</dt><dd className="mono">{(calib.val_fpr_at_threshold * 100).toFixed(1)}% (TPR {(calib.val_tpr_at_threshold * 100).toFixed(1)}%)</dd><dt>val images</dt><dd className="mono">{calib.n_val} ({calib.n_val_real} real)</dd></dl>}
              {hist.length > 0 && (
                <table className="table"><thead><tr><th>epoch</th><th className="num">train loss</th><th className="num">val AUC</th><th className="num">val acc</th><th className="num">attr acc</th></tr></thead>
                  <tbody>{hist.map((h: Any) => <tr key={h.epoch}><td>{h.epoch}</td><td className="num">{h.train_loss.toFixed(4)}</td><td className="num">{h.val_auc.toFixed(4)}</td><td className="num">{(h.val_acc * 100).toFixed(1)}%</td><td className="num">{(h.val_attr_acc * 100).toFixed(1)}%</td></tr>)}</tbody></table>
              )}
              {m.calibration?.reliability && (
                <div><div className="eyebrow" style={{ marginBottom: 6 }}>reliability (test) · bar = observed AI fraction per confidence bin</div>
                  <div style={{ display: 'flex', gap: 3, alignItems: 'flex-end', height: 70 }}>{m.calibration.reliability.map((b: Any, i: number) => <div key={i} title={`${b.bin[0]}–${b.bin[1]}: n=${b.count}`} style={{ flex: 1, height: `${(b.fraction_ai ?? 0) * 100}%`, background: 'var(--amber)', opacity: b.count ? 0.9 : 0.15, borderRadius: 2 }} />)}</div>
                  <div className="row mono small muted" style={{ justifyContent: 'space-between' }}><span>0.0</span><span>predicted P(AI)</span><span>1.0</span></div></div>
              )}
            </div>
          </div>
        </div>
      </>)}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="panel"><div className="panel-h"><h3>Architecture &amp; protocol</h3></div>
          <div className="panel-b stack small" style={{ gap: 10 }}>
            <div><b>Dual-stream detector.</b> RGB stream: ImageNet-pretrained EfficientNet-B0 (timm). Residual stream: three fixed SRM high-pass filters → small CNN, capturing generator noise fingerprints that transfer across generator families. Fusion head outputs the real/AI logit and a generator-attribution logit (multi-task).</div>
            <div><b>Trained for the wild.</b> Random JPEG (q30–95), rescaling, blur, noise, screenshot simulation, colour jitter and crops during training. Both classes are stored with identical resizing and JPEG q95 so file format/resolution cannot be a shortcut.</div>
            <div><b>Calibrated.</b> Temperature scaling on validation; operating point chosen for a {(card.protocol.target_fpr * 100).toFixed(0)}% validation false-positive rate; an explicit “inconclusive” band of ±{(card.protocol.uncertain_band / 2).toFixed(3)} around the threshold.</div>
            <div><b>Unseen-generator protocol.</b> Held out entirely: {card.protocol.held_out_generators.join(', ')}. Seen: {card.protocol.seen_generators.join(', ')}. Attribution classes: {card.protocol.attribution_classes.join(', ')}.</div>
          </div>
        </div>
        <div className="panel"><div className="panel-h"><h3>Datasets &amp; limitations</h3></div>
          <div className="panel-b stack small" style={{ gap: 10 }}>
            {ds ? ds.sources.map((s: Any) => <div key={s.name}><b>{s.name}</b> — {s.role}. {s.size}. Licence: {s.license}. <a href={s.url} target="_blank" rel="noreferrer">source</a></div>) : (<>
              <div><b>CIFAKE</b> (Hugging Face mirror of the Kaggle dataset; real = CIFAR-10, fake = Stable Diffusion 1.4; 32 px). MIT licence.</div>
              <div><b>Tiny-GenImage</b> (curated subset of GenImage: real ImageNet photos + ADM, BigGAN, GLIDE, Midjourney, SD1.4, SD1.5, VQDM, Wukong). CC BY-NC-SA 4.0.</div>
              <div><b>Imagenette</b> (fast.ai, 10 ImageNet classes, 320 px) as additional real photographs. Apache 2.0.</div>
            </>)}
            <div className="divider" />
            <div><b>Known limitations.</b> Trained on a handful of generator families; newer models (Flux, Imagen 3, GPT-image) are untested and may evade detection. Heavy re-compression and small crops reduce reliability. Face-swap deepfakes are out of scope by design. Metadata can be forged or stripped. All outputs are likelihoods, never proof.</div>
          </div>
        </div>
      </div>
    </div>
  )
}
