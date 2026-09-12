"""Generate report/model_report.md (one page), refresh the README metrics block, and render explanation samples.

Usage:
  python -m model.make_report                 # report + README block
  python -m model.make_report --samples 12    # also render report/explanation_samples/ (needs weights)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from datetime import date
from pathlib import Path

from model import config as C


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def pc(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def build_tables(m: dict, calib: dict | None, rob: dict | None, atk: dict | None) -> tuple[str, str]:
    g = m["groups"]
    rows = ["| split | n (real / AI) | ROC-AUC | macro-F1 | accuracy | FPR | TPR |", "|---|---|---|---|---|---|---|"]
    for k in ["unseen", "seen", "highres_all", "cifake", "overall"]:
        if k in g:
            r = g[k]
            name = {"unseen": "**unseen generators** (Midjourney, VQDM)", "seen": "seen generators", "highres_all": "all high-res", "cifake": "CIFAKE test (32 px)", "overall": "overall test"}[k]
            rows.append(f"| {name} | {r['n']} ({r['n_real']} / {r['n_fake']}) | **{fmt(r['auc'])}** | {fmt(r['macro_f1'], 3)} | {pc(r['accuracy'])} | {pc(r['fpr'])} | {pc(r['tpr'])} |")
    core = "\n".join(rows)
    cm = g["overall"]["confusion_matrix"]
    cmu = g["unseen"]["confusion_matrix"] if "unseen" in g else None
    conf = f"""
Confusion matrix @ threshold {m['threshold']:.2f} (rows = truth, cols = prediction):

| overall | pred real | pred AI |
|---|---|---|
| real | {cm['tn']} | {cm['fp']} |
| AI-generated | {cm['fn']} | {cm['tp']} |
"""
    if cmu:
        conf += f"""
| unseen split | pred real | pred AI |
|---|---|---|
| real | {cmu['tn']} | {cmu['fp']} |
| AI-generated | {cmu['fn']} | {cmu['tp']} |
"""
    per = ["| generator | family | held-out | n | AUC vs real | detection rate |", "|---|---|---|---|---|---|"]
    for gname, r in sorted(m["per_generator"].items(), key=lambda kv: -kv[1]["auc"]):
        per.append(f"| {gname} | {r['family']} | {'**yes**' if r['held_out'] else 'no'} | {r['n_fake']} | {fmt(r['auc'])} | {pc(r['detection_rate'], 0)} |")
    per_s = "\n".join(per)
    attr = ""
    if "attribution" in m:
        a = m["attribution"]
        attr = f"Generator attribution (Bonus B, seen test fakes, n={a['n']}): accuracy **{pc(a['accuracy'])}**, macro-F1 {fmt(a['macro_f1'], 3)}, family-level accuracy {pc(a['family_accuracy'])}."
        if a.get("unseen_mapping"):
            attr += " Unseen generators were attributed to: " + "; ".join(f"{k} → " + ", ".join(f"{g2} {n}" for g2, n in sorted(v.items(), key=lambda kv: -kv[1])[:2]) for k, v in a["unseen_mapping"].items()) + "."
    cal = ""
    if calib:
        cal = f"Calibration: temperature T = {calib['temperature']:.3f}; ECE {calib['ece_before']:.4f} → {calib['ece_after']:.4f} on validation; test ECE {m['calibration']['ece']:.4f}. Operating point: threshold {m['threshold']:.2f} chosen for {pc(calib['target_fpr'], 0)} validation FPR (achieved {pc(calib['val_fpr_at_threshold'])}, TPR {pc(calib['val_tpr_at_threshold'])})."
    robs = ""
    if rob:
        rr = rob["results"]
        robs = "\n".join(["| degradation | AUC seen | AUC unseen | accuracy | FPR | verdict flips |", "|---|---|---|---|---|---|"] +
                         [f"| {r['label']} | {fmt(r['auc_seen'], 3)} | {fmt(r['auc_unseen'], 3)} | {pc(r['accuracy'])} | {pc(r['fpr'])} | {pc(r['verdict_flip_rate'])} |" for r in rr.values()])
    atks = ""
    if atk:
        atks = "\n".join(["| attack (white-box) | detection | +JPEG75 | +TTA | +both |", "|---|---|---|---|---|"] +
                         [f"| {k} | {pc(r['plain'])} | {pc(r['jpeg75'])} | {pc(r['tta'])} | {pc(r['jpeg75+tta'])} |" for k, r in atk["detection_rate"].items()])
        atks += "\n\nFPR on real images under each mitigation: " + ", ".join(f"{k} {pc(v)}" for k, v in atk["fpr_on_reals"].items()) + "."
    readme_block = f"""Local held-out test split, threshold {m['threshold']:.2f} (5% validation FPR operating point), temperature-calibrated.

{core}
{conf}
Per generator (fakes vs high-res test reals):

{per_s}

{attr}

{cal}
"""
    full = readme_block + ("\n**Robustness (Bonus C)**\n\n" + robs if robs else "") + ("\n\n**Active defence (Bonus G)**\n\n" + atks if atks else "")
    return readme_block, full


def write_model_report(m, calib, rob, atk, hist, ds) -> Path:
    cfg = m.get("config", {})
    readme_block, full = build_tables(m, calib, rob, atk)
    n_train = cfg.get("n_train", "?")
    epochs = len(hist) if hist else cfg.get("epochs", "?")
    best = max(hist, key=lambda h: h["val_auc"]) if hist else None
    text = f"""# SignalScope — Model Report (one page)

_Generated {date.today().isoformat()} by `python -m model.make_report`._

| Field | Statement |
|---|---|
| **Task** | Binary real vs AI-generated image classification (core). Bonus: A faithful explanation (Grad-CAM + measurable cues), B generator attribution (7-way + family), C robustness to degradation, D provenance/metadata, E caption consistency, F deployable UI, G active-defence analysis. |
| **Data & split** | {'; '.join(f"{s['name']}: {s['role']} ({s['size']}; {s['license']})" for s in ds['sources']) if ds else 'see report/dataset_card.json'}. Train {n_train} images / val {calib['n_val'] if calib else '?'} / test {m['n_test']}. **Held-out generators {', '.join(m['held_out_generators'])} never appear in train/val/calibration.** High-res images stored uniformly (short side 256, JPEG q95) for both classes to remove format/resolution shortcuts. Official SignalScope set: not yet released at build time; adapter ready. |
| **Model / approach** | Dual-stream: EfficientNet-B0 (ImageNet-pretrained, timm) on RGB + fixed SRM high-pass residual stream (3 kernels → 4-block CNN, 128-d) → 256-d fusion → real/AI logit + 7-way generator logits (multi-task, weight {C.ATTR_LOSS_WEIGHT}). Input 224², whole-image resize, hflip TTA. AdamW lr {cfg.get('lr', C.DEFAULT_LR)}, wd {C.WEIGHT_DECAY}, one-cycle cosine, batch {cfg.get('batch', C.DEFAULT_BATCH)}, {epochs} epochs, AMP, label smoothing {C.LABEL_SMOOTHING}, real/fake + per-source balanced sampling. Augmentation: random-resized crop, flip, colour jitter, grayscale, JPEG q30–95, down/up-scaling, blur, noise, screenshot simulation. Calibration: temperature scaling on val; threshold at 5% val FPR; explicit inconclusive band ±{C.UNCERTAIN_BAND / 2:.3f}. Best epoch {best['epoch'] if best else '?'} (val AUC {fmt(best['val_auc']) if best else '?'}). |
| **Metric & result** | See tables below. Primary: **unseen-split AUC {fmt(m['groups'].get('unseen', {}).get('auc'))}**, overall AUC {fmt(m['groups']['overall']['auc'])}; macro-F1 / accuracy / FPR at threshold {m['threshold']:.2f} in the table; confusion matrices below. |
| **Baseline** | The organisers' baseline was not available at build time. Internal baselines: chance AUC 0.5; a CIFAKE-only 32 px model does not transfer to high-res images (AUC ≈ 0.5–0.6 on GenImage in our smoke tests). Our numbers on the unseen split are the ones to compare against the published bands once released. |
| **Limitations** | Trained on six 2021–2023 generator families with a few thousand high-res fakes; newest generators (Flux, Imagen 3, GPT-image, Midjourney v6+) untested. Degradation table shows where accuracy drops (heavy JPEG, 0.25× downscale, screenshots). White-box PGD at 2–4/255 flips many verdicts; JPEG pre-filtering only partially mitigates. Metadata is forgeable; C2PA signatures are not cryptographically verified. Real photos in the test split are ImageNet-domain; portraits/documents are under-represented. Face-swap deepfakes are out of scope. |

## Results

{full}
"""
    out = C.REPORT_DIR / "model_report.md"
    out.write_text(text, encoding="utf-8")
    # README block
    readme = C.ROOT / "README.md"
    if readme.exists():
        s = readme.read_text(encoding="utf-8")
        s = re.sub(r"<!-- METRICS:BEGIN -->.*?<!-- METRICS:END -->", "<!-- METRICS:BEGIN -->\n" + readme_block + "\n<!-- METRICS:END -->", s, flags=re.S)
        readme.write_text(s, encoding="utf-8")
    return out


def render_samples(n: int, seed: int = C.SEED) -> Path:
    """Explanation samples for the Module A rubric: image + heat-map overlay + text + ground truth."""
    import numpy as np
    from PIL import Image
    from model.data.datasets import load_manifest
    from model.predict import Detector
    det = Detector()
    rng = np.random.default_rng(seed)
    test = load_manifest(split="test")
    pools = {
        "real (high-res)": [r for r in test if r["label"] == 0 and r["source"] != "cifake"],
        "seen generator": [r for r in test if r["label"] == 1 and r.get("protocol") == "seen"],
        "unseen generator": [r for r in test if r["label"] == 1 and r.get("protocol") == "unseen"],
    }
    out_dir = C.REPORT_DIR / "explanation_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    lines = ["# Explanation samples", "", f"Random test images (seed {seed}), never used for training. Each row: ground truth, verdict, calibrated P(AI), localisation, cues fired, and the full explanation. Overlays are Grad-CAM of the AI-generated logit.", ""]
    k = 0
    per = max(1, n // len(pools))
    for pool_name, rows in pools.items():
        idx = rng.permutation(len(rows))[:per]
        for i in idx:
            r = rows[i]
            p = C.PROCESSED_DIR / r["path"]
            raw = p.read_bytes()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            res = det.analyze(img, raw_bytes=raw, heatmap=True, tta=True)
            overlay = Image.open(io.BytesIO(base64.b64decode(res["heatmap_png_base64"]))).convert("RGBA")
            side = Image.new("RGB", (img.width * 2 + 8, img.height), (20, 20, 22))
            side.paste(img, (0, 0))
            side.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (img.width + 8, 0))
            fn = f"sample_{k:02d}_{'real' if r['label'] == 0 else r['generator']}.png"
            side.save(out_dir / fn)
            k += 1
            correct = (res["verdict"] == "ai_generated") == (r["label"] == 1)
            lines += [f"## {fn}", "",
                      f"- **Ground truth:** {'real photo' if r['label'] == 0 else 'AI-generated (' + r['generator'] + ', ' + pool_name + ')'}",
                      f"- **Verdict:** {res['label']} — P(AI) = {res['probability_ai']:.3f} ({'correct' if correct else 'WRONG'}), band `{res['band']}`",
                      f"- **Localisation:** {res['regions']['concentration']}, peak regions {', '.join(res['regions']['top_regions'])} ({res['regions']['top2_share'] * 100:.0f}% of heat-map mass)",
                      f"- **Cues fired:** {', '.join(res['explanation']['fired_cues']) or 'none'}",
                      f"- **Attribution:** {res['attribution']['family']} / {res['attribution']['generator']} ({res['attribution']['confidence'] * 100:.0f}%)" if r["label"] == 1 else "- **Attribution:** n/a (verdict real)",
                      "", f"![{fn}]({fn})", ""] + [f"> {s}" for s in res["explanation"]["sentences"]] + [""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0)
    a = ap.parse_args()
    m = _load(C.METRICS_PATH)
    if not m:
        raise SystemExit("report/metrics.json missing - run python -m model.evaluate first")
    out = write_model_report(m, _load(C.CALIBRATION_PATH), _load(C.REPORT_DIR / "robustness.json"), _load(C.REPORT_DIR / "attacks.json"),
                             _load(C.WEIGHTS_DIR / "train_history.json"), _load(C.REPORT_DIR / "dataset_card.json"))
    print(f"model report -> {out}")
    if a.samples:
        print(f"samples -> {render_samples(a.samples)}")


if __name__ == "__main__":
    main()
