# SignalScope - Implementation Plan

Problem: classify an image as **real** vs **AI-generated**, report honest metrics on a held-out set that
includes *unseen generators*, and explain the verdict faithfully. Presented as a likelihood, never an accusation.

Guiding principle: simple, reliable, fast, actually working. One trained model, one API, one database,
one polished UI. Every listed feature is real and verified end-to-end.

---

## 1. Architecture (single machine, three processes)

```
 React/Vite SPA  --HTTP/JSON-->  FastAPI backend  -->  SQLite (SQLAlchemy)
 (app/frontend)                  (app/backend)         analyses, batches, robustness runs
                                       |
                                       v
                               model/ package (PyTorch)
                    detector . calibration . Grad-CAM . cue analysis
                    EXIF/C2PA . robustness . CLIP caption check
```

* `model/` is a standalone Python package with its own CLI (`predict.py`) - the "predict interface"
  organisers call. The backend imports the same code, so the UI and the judged metric share one path.
* The static frontend build is served by FastAPI in production (`/`); the Vite dev server proxies `/api` in dev.

## 2. ML / CV approach

**Backbone:** `timm` EfficientNet-B0 (ImageNet-pretrained, 5.3M params, 224 px) - fast on an RTX 3050 6 GB
and strong under transfer learning. Fine-tuned end-to-end with AdamW, cosine LR, label smoothing, AMP.

**Forensic dual-stream (the "innovation", kept cheap):**

* Stream 1 - RGB semantics/texture: EfficientNet-B0 features (1280-d).
* Stream 2 - Noise-residual / high-frequency stream: image minus its median-blur (SRM-style high-pass),
  through a small 4-block CNN (~0.3M params) -> 128-d. Generator artefacts live in high frequencies and
  transfer across generator families better than semantics do.
* Fusion head -> two outputs: **real/fake logit** and **generator-family logits** (multi-task; the
  attribution head regularises the shared features and powers Bonus B).

**Generalisation & robustness (by construction):** heavy augmentation that mimics the wild:
random JPEG re-compression (q 30-95), random down/up-scaling, Gaussian blur, mild noise, random crops,
colour jitter, horizontal flip, occasional grayscale, "screenshot" simulation (resize + re-encode).

**Calibration:** temperature scaling fitted on the validation split; confidence is reported after calibration,
with an explicit "uncertain" band around the operating threshold.

## 3. Data & training pipeline (replaceable)

Manifest-driven: `data/manifest.csv` with `path,label,generator,source,split`. Adapters in
`model/data/prepare.py` convert each raw source into files + manifest rows. Adding the official SignalScope
set = one adapter function + one CLI flag.

| Source | Role | Licence |
|---|---|---|
| CIFAKE (HF `dragonintelligence/CIFAKE-image-dataset`) 100k/20k, 32 px | core "CIFAKE-style" train/val/test | MIT (CIFAKE); CIFAR-10 real photos |
| Tiny-GenImage (HF `TheKernel01/Tiny-GenImage`) 28k/7k, 8 generators | high-res train + **unseen-generator** protocol + attribution | CC BY-NC-SA 4.0 (GenImage) |

**Unseen-generator protocol (honest):** generators split into *seen* (SD1.4, SD1.5, ADM, GLIDE, VQDM, Wukong)
and *held-out* (Midjourney, BigGAN - one diffusion, one GAN). Held-out generator images are **never** used for
training or calibration; they form the local "unseen split" on which AUC is reported separately.
Real images in the test split are disjoint from training. Nothing from the organisers' held-out set is ever touched.

Steps (all `python -m model.<step>`): `data.download` -> `data.prepare` (decode, uniform resize to 320 px
short side, JPEG q95 for *both* classes so compression cannot become a shortcut, write manifest) -> `train`
-> `calibrate` -> `evaluate` (ROC-AUC overall / seen / unseen, macro-F1, confusion matrix, accuracy & FPR at
threshold, per-generator AUC, degradation curves, attribution accuracy) -> `report/` artefacts.

Training budget: ~70k images x 224 px, 6-8 epochs, AMP, batch 64 -> roughly 1 h on the RTX 3050.

## 4. Explainability (Module A - faithful, localised, hedged)

* **Grad-CAM** on the last conv block of the RGB stream, targeting the "AI-generated" logit -> heat-map.
  For a "likely real" verdict the low-activation map is shown with wording that no strong synthetic cue was found.
* **Grounded cue analysis** - measurable image statistics, each reported with its value and what is typical:
  high-frequency energy / spectral slope, noise-residual uniformity across regions (camera sensor noise is
  spatially consistent; generators produce over-smooth or patchy residuals), JPEG grid / double-compression
  hints, colour-channel correlation, saturation extremes, local texture repetition, edge sharpness profile.
* **Explanation text** is assembled from *only* the cues that actually fired, ranked by strength, joined
  with the heat-map peak regions ("strongest signal in the upper-left quadrant"), with uncertainty phrasing
  driven by the calibrated probability. No free-text LLM hallucination: templated, verifiable statements.

## 5. Bonus modules built

* **B Attribution** - generator-family head (GAN vs diffusion + specific family), with per-class metrics.
* **C Robustness** - `robustness.py` evaluates JPEG q in {95,75,50,30}, resize 0.5x/0.25x, blur, screenshot
  simulation -> degradation-vs-AUC table + chart; the UI runs a live "stress test" per image.
* **D Provenance** - EXIF (Pillow/piexif), C2PA manifest detection (JUMBF box scan in JPEG/PNG), software
  tags (e.g. "Stable Diffusion", "Midjourney"), generator markers in metadata. Fused with the visual verdict
  by explicit rules: metadata can *raise* suspicion or *corroborate*, never silently override; shown as a
  separate evidence lane.
* **E Multimodal** - OpenCLIP ViT-B/32 image-caption similarity -> "consistent / weak / inconsistent" band
  (generic captions only).
* **F Deployable** - drag-and-drop single scan, batch scan, analysis history, comparison, JSON/HTML report
  export, sub-second latency on GPU.
* **G Active defence** - adversarial robustness study: FGSM/PGD small-epsilon attacks + post-processing attacks
  on the eval set, with mitigation (JPEG-aware training, test-time averaging) results in the report.

## 6. Backend / API (FastAPI)

`POST /api/analyze` (multipart image + optional caption) -> full result JSON;
`POST /api/analyze/batch`; `GET /api/analyses` (history, paging, filters); `GET /api/analyses/{id}`;
`DELETE /api/analyses/{id}`; `POST /api/analyses/{id}/robustness`; `GET /api/analyses/{id}/report`;
`GET /api/stats`; `GET /api/model` (metrics, thresholds, dataset card); `GET /api/health`.
Pydantic validation, file-type/size limits, structured error responses, request timing.

## 7. Database (SQLite via SQLAlchemy 2)

Tables: `analyses` (id, filename, sha256, dims, verdict, probability, calibrated confidence, band,
attribution JSON, cues JSON, metadata JSON, caption/consistency, heat-map path, latency, created_at, batch_id),
`batches`, `robustness_runs`. Files (originals, heat-maps) live in `app/backend/storage/`.

## 8. Frontend (React + Vite + TypeScript, hand-written CSS, no UI kit)

Identity: "forensic lab bench" - deep graphite background, paper-white evidence cards, amber
"signal" accent, monospace data readouts, scan-line / reticle motifs, subtle film grain. Not a SaaS dashboard.
Views: **Scan** (drop zone -> evidence report with verdict dial, heat-map overlay slider, cue ledger,
provenance lane, attribution, stress test), **Batch**, **Case files** (history with filters, compare two),
**Model card** (metrics, ROC, confusion matrix, degradation curves, limitations). Responsive, keyboard
accessible, reduced-motion aware.

## 9. Testing

`pytest`: model forward/shape + determinism, preprocessing, cue extraction, calibration math, EXIF/C2PA
parser on synthetic files, API endpoints (TestClient with a tiny image), DB CRUD, report export.
Frontend: type-check + production build.

## 10. Run / deployment

`scripts/setup.ps1|sh` (install deps, download weights), `scripts/run.ps1|sh` (backend + frontend),
`python -m model.predict image.jpg` CLI, Dockerfile (CPU) for judges. Weights published via GitHub release;
`weights/README.md` links. README with reproduction in under 10 minutes, metrics, datasets/licences,
originality declaration. `report/model_report.md` one-pager + explanation samples.
