# SignalScope — Telling Real From Synthetic

> Media-forensics tool that decides whether an image is **real or AI-generated**, reports honest metrics on a held-out
> split that includes **generators never seen in training**, and explains each verdict with a **Grad-CAM evidence map
> and measurable forensic cues** — presented as a likelihood, never an accusation.

SIH 2026 internal hackathon · Problem Statement 2 · L. J. Institute of Engineering and Technology

---

## 1. What was built

| | Module | Status | Where |
|---|---|---|---|
| **Core** | Real vs AI-generated classification, calibrated confidence, ROC-AUC / macro-F1 / confusion matrix on a held-out set with an unseen-generator split, predict interface (CLI + API + UI) | ✅ | `model/`, `app/` |
| **A** | Faithful explanation: Grad-CAM heat-map of the AI logit, 3×3 localisation, 9 measurable cues compared against real-photo reference ranges, templated hedged text | ✅ headline | `model/explain.py`, `model/cues.py` |
| **B** | Generator attribution (7-way head; family-level roll-up: GAN / pixel-diffusion / latent-diffusion / VQ-diffusion) | ✅ | `model/nets.py`, `model/evaluate.py` |
| **C** | Robustness to degradation: JPEG q30–90, down-scaling, blur, noise, screenshot; offline study + per-image live stress test | ✅ | `model/robustness.py` |
| **D** | Provenance & metadata: EXIF, XMP/PNG text, generator markers, C2PA manifest presence, explicit fusion rule | ✅ | `model/provenance.py` |
| **E** | Image–caption consistency (OpenCLIP ViT-B/32) | ✅ | `model/multimodal.py` |
| **F** | Deployable UI: drag-and-drop scan, batch scan (≤50), case files with filters, side-by-side compare, HTML/JSON evidence reports, reviewer decisions | ✅ | `app/frontend`, `app/backend` |
| **G** | Active-defence analysis: FGSM/PGD white-box attacks + JPEG/TTA mitigations, honest failure table | ✅ | `model/attacks.py` |

Everything listed is implemented and exercised by `pytest` (23 tests) and the demo.

## 2. Quick start (judge path, < 10 minutes)

Prerequisites: Python 3.10–3.12, Node 18+, (optional) NVIDIA GPU.

```bash
git clone <this repo> SignalScope && cd SignalScope
# Windows
.\scripts\setup.ps1
.\scripts\run.ps1
# Linux / macOS
bash scripts/setup.sh
bash scripts/run.sh
```

Then open **http://localhost:8000**. The setup script installs PyTorch (CUDA if `nvidia-smi` is present, else CPU),
Python and Node dependencies, builds the frontend and fetches the released weights into `weights/`
(`python scripts/get_weights.py`; see `weights/README.md`).

Predict on a single new image from the command line (the required *predict interface*):

```bash
python -m model.predict path/to/image.jpg --heatmap out.png --json out.json
```

Batch-predict a folder (one CSV row per image: `filename, p_ai_generated, label, threshold`) — this is the entry
point organisers can run on their held-out set:

```bash
python -m model.predict --dir path/to/folder --csv predictions.csv
```

Run the tests: `python -m pytest tests -q` (no weights or network required — tests build a tiny checkpoint).

Docker (CPU): `docker build -t signalscope . && docker run -p 8000:8000 signalscope`.

### Manual setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -r requirements.txt
cd app/frontend && npm install && npm run build && cd ../..
python scripts/get_weights.py            # or train (section 4)
uvicorn app.backend.main:app --port 8000
```

Dev mode with hot reload: `scripts/run.ps1 -Dev` / `bash scripts/run.sh --dev` (API on 8010, Vite on 5173).

## 3. Datasets & licences

The official SignalScope dataset had not been released when this was built, so the pipeline trains on public data
and is **manifest-driven** so the official set drops in without code changes elsewhere
(`model/data/prepare.py::adapt_signalscope`, expects `data/raw/signalscope/{train,val,test}/{real,fake}/`).

| Dataset | Use | Size used | Licence / source |
|---|---|---|---|
| **CIFAKE** (HF mirror `dragonintelligence/CIFAKE-image-dataset`) | "CIFAKE-style" core set; real = CIFAR-10, fake = Stable Diffusion 1.4, 32 px | 20k+20k train, 1k+1k val, 10k+10k test | MIT · Bird & Lotfi 2024 |
| **Tiny-GenImage** (`TheKernel01/Tiny-GenImage`, subset of GenImage) | high-res real ImageNet photos + fakes from ADM, BigGAN, GLIDE, Midjourney, SD1.4, SD1.5, VQDM, Wukong | see `report/metrics.json → config.n_train` | CC BY-NC-SA 4.0 · Zhu et al. 2023 |
| **Imagenette** 320 px (fast.ai) | additional real photographs (same ImageNet domain) to balance real/fake | 9,469 train, 3,925 val | Apache 2.0 |

No images of identifiable individuals were sourced. Full card: `report/dataset_card.json`.

**Unseen-generator protocol.** `Midjourney` and `VQDM` are *held out entirely* — never used for training,
validation, calibration or cue-reference fitting. They form the local **unseen split**. The other six generators are
"seen". The organisers' held-out set is never touched; our reported core metric is what `python -m model.evaluate`
computes on our own test split, and the organisers compute theirs via the predict interface above.

**Shortcut removal.** All high-res images (both classes) are stored at short side 256 as JPEG q95, so file format and
resolution — which differ per generator in raw GenImage — cannot be learned instead of forensics.

## 4. Reproduce training (≈1 h on an RTX 3050 6 GB)

```bash
python -m model.data.download            # CIFAKE + Tiny-GenImage shards from Hugging Face
#   Imagenette: download https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and extract to data/raw/imagenette/
python -m model.data.prepare             # -> data/processed + data/manifest.csv
python -m model.train --epochs 8         # -> weights/signalscope_best.pt (best val AUC)
python -m model.calibrate                # temperature scaling + FPR-targeted threshold -> weights/calibration.json
python -m model.cues fit                 # real-photo reference ranges for the cue ledger
python -m model.evaluate                 # -> report/metrics.json  (AUC overall/seen/unseen, F1, confusion, per generator)
python -m model.robustness               # -> report/robustness.json
python -m model.attacks                  # -> report/attacks.json
```

## 5. Results (local held-out test split)

<!-- METRICS:BEGIN -->
_Fill in by running `python -m model.evaluate`; see `report/model_report.md`._
<!-- METRICS:END -->

Per-generator AUC, degradation curves, adversarial results, calibration and training history are shown live in the
app's **Model card** view and stored under `report/`.

## 6. Architecture

```
image (+caption, +bytes) ─► preprocessing (whole-image 224², hflip TTA)
        ├─► RGB stream: EfficientNet-B0 (ImageNet-pretrained, timm)            ─┐
        └─► residual stream: 3 fixed SRM high-pass filters → 4-block CNN        ─┴► fusion(256) ─► real/AI logit
                                                                                              └► generator logits (7)
logit ─► temperature scaling ─► calibrated P(AI) ─► band {likely real · possibly real · inconclusive · possibly AI · likely AI}
        ├─► Grad-CAM on the RGB stream's last conv (AI logit) → heat-map, 3×3 mass, peak bbox
        ├─► 9 measurable cues vs real-photo 5–95th percentile reference  → "fired" cues
        ├─► EXIF / XMP / PNG-text / C2PA scan → provenance signal → explicit fusion rule
        └─► OpenCLIP caption agreement (optional)
                     ─► templated, hedged explanation citing only cues that fired
```

* **Why a residual stream?** Generator fingerprints live in high-frequency noise, which transfers across generator
  families better than semantics — this is the generalisation bet for the unseen split.
* **Trained for the wild.** Random JPEG (q30–95), rescaling, blur, noise, screenshot simulation, colour jitter,
  random-resized crops. Real/fake and per-source balanced sampling.
* **Calibration & operating point.** Temperature fitted on validation; threshold chosen for a 5% validation
  false-positive rate (flagging a real photo is the costly error). An explicit *inconclusive* band around the threshold
  is surfaced in the UI instead of a forced answer.
* **Faithfulness guard-rails.** Text is assembled only from cues whose measured value is outside the real-photo range,
  localisation statements come from the heat-map mass distribution, and wording strength is tied to the calibrated
  probability. No free-form LLM text.
* **Provenance fusion.** An explicit generator marker raises the combined likelihood to ≥0.90; camera EXIF can
  corroborate but never lowers the visual likelihood (it is trivially copied); missing metadata is neutral.

## 7. Known limitations (honest)

* Trained on a handful of 2021–2023 generator families at modest scale; newer models (Flux, Imagen 3, GPT-image,
  Midjourney v6+) are untested and may evade detection. Report `unseen` AUC is the number to trust, not `seen`.
* Heavy re-compression, tiny crops and screenshots reduce reliability (quantified in the robustness table).
* White-box adversarial perturbations of a few grey levels can flip verdicts; JPEG pre-filtering only partially helps.
* Metadata can be forged or stripped; C2PA signatures are detected but not cryptographically verified.
* Face-swap deepfakes of real people are out of scope by design; the tool makes no claims about individuals or events.

## 8. Repository layout

```
model/        ML package: data/ (download, prepare, datasets), nets.py, train.py, calibrate.py, evaluate.py,
              predict.py (predict interface), explain.py, cues.py, provenance.py, robustness.py, attacks.py, multimodal.py
app/backend   FastAPI + SQLAlchemy/SQLite (analyses, batches, reviews, reports, stats)
app/frontend  Vite + React + TypeScript UI
report/       metrics.json, robustness.json, attacks.json, dataset_card.json, model_report.md, explanation_samples/
weights/      checkpoint + calibration (via release or training)
tests/        pytest suite (model, explanation, provenance, API)
scripts/      setup/run scripts, weight download
```

API docs: `http://localhost:8000/docs`.

## 9. Originality declaration

All code in this repository was written for this hackathon during the event window. Third-party components used
(all cited, none copied as notebooks): PyTorch, `timm` (EfficientNet-B0 ImageNet weights), OpenCLIP (ViT-B/32 OpenAI
weights), FastAPI, SQLAlchemy, React/Vite, scikit-learn, Pillow, NumPy. The SRM high-pass kernels are the standard
filters from Fridrich & Kodovský (2012) as popularised in image-forensics literature; Grad-CAM follows Selvaraju et
al. (2017) and is implemented from scratch in `model/explain.py`. Datasets: CIFAKE, GenImage (Tiny-GenImage subset),
Imagenette — see section 3. AI coding assistance was used during development.

## 10. Demo video

_Link to the 3–5 minute demo: TBD_
