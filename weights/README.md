# Weights

This folder holds the trained detector and its calibration:

| file | purpose |
|---|---|
| `signalscope_best.pt` | SignalScopeNet checkpoint (EfficientNet-B0 RGB stream + SRM residual stream + fusion/attribution heads), ~19 MB |
| `calibration.json` | temperature `T`, operating threshold, validation FPR/TPR |
| `cue_reference.json` | 5th/95th percentile ranges of every forensic cue measured on real validation photos |
| `train_history.json` | per-epoch training log |

Large binaries are not committed. Get them either way:

1. **Download the release** (fastest): `python scripts/get_weights.py` reads `weights/release.json` and fetches the files
   from the GitHub release attached to this repository.
2. **Reproduce**: `python -m model.data.download && python -m model.data.prepare && python -m model.train && python -m model.calibrate && python -m model.cues fit && python -m model.evaluate`
   (about 1 h on an RTX 3050 6 GB; see README for details).
