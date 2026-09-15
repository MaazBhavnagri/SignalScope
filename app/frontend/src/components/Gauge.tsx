// Likelihood dial: arc from 0 (real) to 1 (AI) with the decision threshold and the uncertainty band marked.
import type { Band } from '../api'
import { tone } from '../api'

interface Props { p: number; threshold: number; band: Band; size?: number; uncertainBand?: number; light?: boolean }

const polar = (cx: number, cy: number, r: number, a: number) => [cx + r * Math.cos(a), cy + r * Math.sin(a)] as const
const arc = (cx: number, cy: number, r: number, a0: number, a1: number) => {
  const [x0, y0] = polar(cx, cy, r, a0); const [x1, y1] = polar(cx, cy, r, a1)
  return `M ${x0} ${y0} A ${r} ${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1} ${y1}`
}

export default function Gauge({ p, threshold, band, size = 200, uncertainBand = 0.15, light = false }: Props) {
  const cx = size / 2, cy = size / 2 + 8, r = size / 2 - 16
  const A0 = Math.PI * 0.85, A1 = Math.PI * 2.15 // 234° sweep
  const at = (v: number) => A0 + (A1 - A0) * Math.min(1, Math.max(0, v))
  const t = tone(band)
  const color = t === 'ai' ? 'var(--ai)' : t === 'real' ? 'var(--real)' : 'var(--unsure)'
  const trackColor = light ? 'var(--paper-line)' : 'var(--line)'
  const textColor = light ? 'var(--paper-ink)' : 'var(--ink)'
  const subColor = light ? 'var(--paper-ink-2)' : 'var(--ink-3)'
  const lo = Math.max(0, threshold - uncertainBand * 0.5), hi = Math.min(1, threshold + uncertainBand * 0.5)
  const [tx, ty] = polar(cx, cy, r, at(threshold)); const [tx2, ty2] = polar(cx, cy, r + 11, at(threshold))
  const [nx, ny] = polar(cx, cy, r - 22, at(p))
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Likelihood of AI generation ${(p * 100).toFixed(0)} percent`}>
      <path d={arc(cx, cy, r, A0, A1)} stroke={trackColor} strokeWidth={9} fill="none" strokeLinecap="round" />
      <path d={arc(cx, cy, r, at(lo), at(hi))} stroke="var(--unsure)" strokeOpacity={0.25} strokeWidth={9} fill="none" />
      <path d={arc(cx, cy, r, A0, at(p))} stroke={color} strokeWidth={9} fill="none" strokeLinecap="round" style={{ transition: 'all .6s cubic-bezier(.2,.8,.2,1)', filter: 'drop-shadow(0 0 6px currentColor)' }} />
      <line x1={tx} y1={ty} x2={tx2} y2={ty2} stroke={textColor} strokeWidth={2} opacity={0.6} />
      <circle cx={nx} cy={ny} r={3.5} fill={color} style={{ filter: 'drop-shadow(0 0 4px currentColor)' }} />
      <text x={cx} y={cy - 6} textAnchor="middle" fontFamily="var(--mono)" fontSize={size * 0.19} fontWeight={600} fill={textColor}>{(p * 100).toFixed(0)}%</text>
      <text x={cx} y={cy + 16} textAnchor="middle" fontFamily="var(--mono)" fontSize={10} letterSpacing={1.4} fill={subColor}>P(AI-GENERATED)</text>
      <text x={polar(cx, cy, r + 2, A0)[0] - 2} y={polar(cx, cy, r + 2, A0)[1] + 16} fontFamily="var(--mono)" fontSize={9} fill={subColor}>REAL</text>
      <text x={polar(cx, cy, r + 2, A1)[0] + 2} y={polar(cx, cy, r + 2, A1)[1] + 16} textAnchor="end" fontFamily="var(--mono)" fontSize={9} fill={subColor}>AI</text>
      <text x={cx} y={size - 2} textAnchor="middle" fontFamily="var(--mono)" fontSize={9} fill={subColor}>threshold {threshold.toFixed(2)}</text>
    </svg>
  )
}
