# Experiment status

Checked: 2026-11-10

## CVOGL audit

Run `python scripts/audit_cvogl.py --workspace /path/to/workspace` with CVOGL extracted under `data/CVOGL`. Dependencies: torch (supports weights_only=True), numpy, Pillow. The original audit ran on Python 3.11 and PyTorch 2.0.1 CPU. Outputs go to `audit_20261110/`; full image decoding and hashing may take several minutes. These dependencies are not yet a locked inference environment.

SVI splits: train 4343, val 923, test 973. All 11114 SVI images decoded; no missing referenced images or invalid boxes/prompts. SHA-256 query-content overlaps: train/val 10, train/test 17, val/test 5 distinct hashes. There is one shared reference image between val/test. Shared input scenes do not automatically mean identical target objects.

Fixed training-median-box baseline: test IoU>=0.5 = 8/973 (0.822%), mean IoU 0.03438. This is not trained-model performance. Data contains large structural objects, not the planned street-furniture categories. No meter-level accuracy claim is made.

## SVI checkpoints

- [DetGeo](https://github.com/sunyuxi/DetGeo): no SVI checkpoint link confirmed in reviewed README/issues; Releases empty at inspection. This is not proof no checkpoint exists elsewhere.
- [OCGNet](https://github.com/ZheyangH/OCGNet): README explicitly supplies drone-to-satellite checkpoints; SVI evaluation filename is not a download link. Releases empty.
- [CVOS](https://github.com/zqwlearning/CVOS-Code#model-zoo): official Model Zoo explicitly links a Ground-to-Satellite checkpoint: https://drive.google.com/file/d/1MDEpopjDWDpfbDd2Co0osCoWVAcEiGeN/view . Candidate only: not downloaded or loaded. Uses CVOS architecture, not interchangeable with DetGeo/OCGNet. Author additionally provides CVOGL-Seg masks for segmentation evaluation.

Next: obtain and inspect the CVOS checkpoint and exact dependencies, verify batch-size-1 memory use before promising local inference; obtain SVI-specific DetGeo/OCGNet weights separately if reproducing those methods.

Only code, configurations and documented results belong in Git. Datasets and large weights remain local. Upstream code should be tracked by source and revision with applicable license terms.
