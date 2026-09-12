"""Self-contained printable HTML evidence report for one analysis."""
from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from pathlib import Path

from app.backend import models


def _b64(path: str | None) -> str | None:
    if path and Path(path).exists():
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return None


def _mime(path: str | None) -> str:
    ext = (Path(path).suffix.lower() if path else "")
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp", ".tif": "image/tiff"}.get(ext, "image/jpeg")


def render_report_html(a: models.Analysis) -> str:
    e = html.escape
    img = _b64(a.original_path)
    hm = _b64(a.heatmap_path)
    expl = a.explanation or {}
    cues = a.cues or []
    prov = a.provenance or {}
    attr = a.attribution or {}
    rob = a.robustness
    p = a.probability_ai
    ai = a.verdict == "ai_generated"
    color = "#b4501e" if ai else "#2f6f4e"
    if a.band == "uncertain":
        color = "#8a6d1f"

    cue_rows = "".join(
        f"<tr class='{'fired' if c['fired'] else ''}'><td>{e(c['name'])}</td><td class='mono'>{c['value']}</td>"
        f"<td class='mono'>{c['typical_range'][0]} – {c['typical_range'][1]}</td><td>{e(c['finding'])}</td></tr>" for c in cues)
    prov_rows = "".join(f"<li>{e(r)}</li>" for r in prov.get("reasons", []))
    sentences = "".join(f"<li>{e(s)}</li>" for s in expl.get("sentences", []))
    attr_rows = ""
    if ai and attr.get("distribution"):
        attr_rows = "".join(f"<tr><td>{e(g)}</td><td class='mono'>{round(v * 100, 1)}%</td></tr>" for g, v in attr["distribution"].items())
    rob_rows = ""
    if rob:
        rob_rows = "".join(f"<tr><td>{e(i['label'])}</td><td class='mono'>{i['probability_ai']:.3f}</td><td class='mono'>{i['delta']:+.3f}</td><td>{'flipped' if i['flipped'] else 'stable'}</td></tr>" for i in rob["items"])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SignalScope evidence report – {e(a.filename)}</title>
<style>
 body{{font-family:Georgia,'Times New Roman',serif;color:#1d1c1a;background:#f6f3ec;margin:0;padding:32px}}
 .sheet{{max-width:900px;margin:0 auto;background:#fffdf8;border:1px solid #d9d2c2;padding:40px 48px;box-shadow:0 2px 0 #d9d2c2}}
 h1{{font-size:26px;margin:0 0 4px;letter-spacing:.5px}} h2{{font-size:15px;text-transform:uppercase;letter-spacing:.14em;color:#6b6457;margin:28px 0 10px;border-bottom:1px solid #d9d2c2;padding-bottom:6px}}
 .mono{{font-family:'IBM Plex Mono','Consolas',monospace;font-size:13px}} .meta{{color:#6b6457;font-size:13px}}
 .verdict{{display:flex;gap:24px;align-items:center;margin:20px 0;padding:18px 20px;border:2px solid {color};border-radius:4px}}
 .verdict .big{{font-size:22px;font-weight:700;color:{color}}} .verdict .num{{font-size:40px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:{color}}}
 .imgs{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .imgs figure{{margin:0;position:relative}} .imgs img{{width:100%;display:block;border:1px solid #d9d2c2}}
 .imgs .ov{{position:absolute;inset:0;width:100%;height:100%}} figcaption{{font-size:12px;color:#6b6457;margin-top:4px}}
 table{{width:100%;border-collapse:collapse;font-size:13px}} td,th{{padding:6px 8px;border-bottom:1px solid #ece6d8;text-align:left;vertical-align:top}} tr.fired td{{background:#fdf1e3}}
 ul{{padding-left:18px}} li{{margin:4px 0}} .foot{{margin-top:32px;font-size:11px;color:#6b6457;border-top:1px solid #d9d2c2;padding-top:10px}}
 @media print{{body{{background:#fff;padding:0}} .sheet{{border:0;box-shadow:none}}}}
</style></head><body><div class="sheet">
<div class="meta mono">SIGNALSCOPE · EVIDENCE REPORT · CASE {e(a.id)}</div>
<h1>{e(a.filename)}</h1>
<div class="meta">{a.width}×{a.height} px · {a.size_bytes:,} bytes · analysed {e(a.created_at.isoformat() if a.created_at else '')} · sha256 <span class="mono">{e(a.sha256[:24])}…</span></div>
<div class="verdict"><div class="num">{round(p * 100)}%</div><div><div class="big">{e(a.label)}</div><div class="meta">calibrated likelihood of AI generation · decision threshold {a.threshold:.2f} · band: {e(a.band)}</div>
<div class="meta">This is a likelihood assessment produced by a statistical model, not a determination of fact.</div></div></div>
<div class="imgs">
 <figure>{f'<img src="data:{_mime(a.original_path)};base64,{img}">' if img else '<div class="meta">original not stored</div>'}<figcaption>Original</figcaption></figure>
 <figure>{f'<img src="data:{_mime(a.original_path)};base64,{img}">' if img else ''}{f'<img class="ov" src="data:image/png;base64,{hm}">' if hm else ''}<figcaption>Grad-CAM evidence heat-map (AI-generated logit) · {e((a.regions or {}).get('concentration', ''))} · peak regions: {e(', '.join((a.regions or {}).get('top_regions', [])))}</figcaption></figure>
</div>
<h2>Explanation</h2><ul>{sentences}</ul>
<h2>Measured cues</h2><table><tr><th>Cue</th><th>Measured</th><th>Real-photo range (5–95%)</th><th>Finding</th></tr>{cue_rows}</table>
{f"<h2>Generator attribution (hint)</h2><p class='meta'>Most consistent with <b>{e(attr.get('family',''))}</b> · closest known generator {e(attr.get('generator',''))} ({round(attr.get('confidence',0)*100)}%). {e(attr.get('note',''))}</p><table>{attr_rows}</table>" if attr_rows else ''}
<h2>Provenance &amp; metadata</h2><p class="meta">Signal: <b>{e(prov.get('signal', 'n/a'))}</b>{(' · JPEG quality ≈ ' + str(prov.get('jpeg_quality_estimate'))) if prov.get('jpeg_quality_estimate') else ''}</p><ul>{prov_rows}</ul>
{f"<p class='meta'>Fusion rule: {e((a.combined or {}).get('rule',''))}</p>" if a.combined else ''}
{f"<h2>Caption consistency</h2><p>{e((a.caption_consistency or {}).get('sentence',''))}</p>" if a.caption_consistency and a.caption_consistency.get('band') else ''}
{f"<h2>Robustness stress test</h2><p class='meta'>{e(rob.get('note',''))}</p><table><tr><th>Degradation</th><th>P(AI)</th><th>Δ</th><th>Verdict</th></tr>{rob_rows}</table>" if rob else ''}
{f"<h2>Reviewer decision</h2><p><b>{e(a.review_decision or '')}</b> {e(a.review_note or '')}</p>" if a.review_decision else ''}
<div class="foot">Generated by SignalScope 1.0 on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Model: {e(str((a.model_info or {}).get('backbone', '')))} dual-stream detector, temperature-calibrated. Outputs are probabilistic; the system makes no claims about real individuals or events.</div>
</div></body></html>"""
