# Demo video script (3–5 minutes)

Record at 1440×900 or wider. Start the app with `scripts/run.ps1` (http://localhost:8000). Have ready: one real photo
you took yourself (no identifiable people), one AI image saved from a generator (PNG straight from the tool keeps its
metadata), and a JPEG screenshot of that same AI image.

| t | Screen | Say |
|---|---|---|
| 0:00 | Model card view | "SignalScope decides whether an image is real or AI-generated and explains why. It's trained on public CIFAKE and GenImage data; Midjourney and VQDM were held out completely, so the *unseen-generator* AUC you see here — the number the challenge weights most — is measured on generators the model never saw." Point at the unseen AUC tile, the per-generator bars and the confusion matrix. |
| 0:45 | Scan → drop the AI PNG | "Core task: one image in, a calibrated likelihood out." Show the dial, stamp and the hedged wording ("likely", never "certain"). Read the calibrated vs raw numbers. |
| 1:15 | Evidence map | "Module A. The heat-map is Grad-CAM of the AI logit, so it shows where the evidence came from. The reticle marks the peak region; the 3×3 grid shows how concentrated it is." Toggle the grid and the opacity slider. |
| 1:40 | Cue ledger | "Every sentence in the explanation maps to a number here. Cues fire only when they leave the 5th–95th percentile range measured on real photos — so it's verifiable, not eloquent guessing." Point at a fired row and its finding text. |
| 2:05 | Provenance + attribution lanes | "Module D read the PNG text chunk that the generator embedded — explicit metadata raises the combined likelihood, but camera EXIF can never lower it because it is trivially copied. Module B says which generator family the artefacts resemble; unseen generators map to the nearest known family and the UI says so." |
| 2:35 | Stress test | Click *Run stress test*. "Module C: the same image after JPEG q30, downscaling, blur, noise and a screenshot. The verdict is stable/fragile, with the likelihood shift per degradation." |
| 2:55 | Drop the screenshot JPEG | "The real-world case: a re-compressed screenshot. Metadata is gone, the model still leans AI, and the compression caveat appears in the text." |
| 3:20 | Drop the real photo | "A real photo: low likelihood, weak scattered activation, cues inside the real range, and — because it's *your* camera file — EXIF corroborates." Type a generic caption and show the caption consistency lane (Module E). |
| 3:50 | Batch scan | Drop 6–10 mixed images. "Module F: triage a folder, sort by likelihood, every image gets a case file." |
| 4:10 | Case files | Filter, tick two cases → side-by-side comparison. Open an *Evidence report* (print-ready HTML). Mark a reviewer decision. |
| 4:35 | Model card → robustness + attacks tables | "Honest limits: accuracy under heavy degradation and white-box PGD attacks, with the mitigations we measured. Everything here is reproducible from the README in under ten minutes with the released weights." |
| 4:55 | Terminal | `python -m model.predict --dir samples --csv out.csv` — "and this is the predict interface for the organisers' held-out set." |
