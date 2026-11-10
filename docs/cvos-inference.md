# CVOS SVI inference

Validated on 2026-11-10 with NVIDIA RTX 3060 Laptop (6 GiB), Windows, Python 3.11, torch 2.5.1+cu121, torchvision 0.20.1+cu121, einops 0.8.1, numpy, Pillow. Uses the existing GPU environment without modifying it.

## Setup

Clone https://github.com/zqwlearning/CVOS-Code into a separate local directory, then checkout `289602a033d5353d06987a2d56a808e9c938cdad`. The wrapper imports this source without vendoring upstream code. Obtain the author's Ground-to-Satellite checkpoint and CVOGL separately.

```sh
python scripts/evaluate_cvos.py --upstream /path/CVOS-Code --data /path/CVOGL/CVOGL_SVI --checkpoint /path/model_svi_model_best.pth.tar --output /path/results/cvos-smoke --split val --limit 8
```

Use `--limit 0` for a complete split, and `--split test` for test evaluation. Each run should have its own output directory. Outputs: predictions.jsonl, summary.json, first four reference-image overlays (green truth, red prediction). No segmentation metrics are computed. CVOGL-Seg masks are unnecessary for box-only inference.

## Fidelity and safeguards

- Restricted `weights_only=True` deserialization; remove DataParallel prefix only.
- All 736 state entries load strictly, and duplicated shared-backbone tensors must agree.
- Skip redundant ImageNet download only during model construction; the full task checkpoint supplies all parameters.
- Preserve 512x256 ground images, 1024x1024 reference images, original normalization, integer-truncated click coordinates and target boxes.
- Use upstream decoding and IoU, strict `IoU > threshold` at 0.25 and 0.5. Dummy cell IDs affect only an unused center-cell metric.
- FP32 inference, batch size 1, no training or test-time adaptation.

## Verified smoke result

Checkpoint SHA256: `036095a7a65fe5451545a014673e6b1b64c528e40ac9a6ad730a76d69d439e5b`; checkpoint epoch field: 14.

First 8 validation records: mean IoU 0.295799; IoU>0.5 and IoU>0.25 each 3/8. Peak PyTorch CUDA allocated memory 593.25 MiB (not total driver/process memory). First overlay visually inspected; predicted and true building boxes overlap substantially. This is a deterministic smoke subset, not representative accuracy or a paper reproduction claim. Full validation/test runs and stratified results remain pending.
