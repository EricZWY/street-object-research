"""Grounding DINO-T baseline on the frozen Vistas development split.

Same 56 development images, same 844 ground-truth boxes, same 13-class label
space and the same full-image + 1024/768 tiling geometry as the YOLO-World
matrix, so the resulting AP50 is directly comparable. Nothing is trained here:
this is an inference-only zero-shot baseline.

The checkpoint is consumed through transformers (GroundingDinoForObjectDetection)
from weights/GDino_T_ds, which avoids building the CUDA MultiScaleDeformableAttention
extension on Windows. A newer transformers lives in deps_ds and is put in front
of sys.path for this process only.
"""
import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms

ap = argparse.ArgumentParser()
ap.add_argument("--workspace", type=Path, required=True)
ap.add_argument("--run-name", required=True)
ap.add_argument("--prompt-set", default="original")
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--nms", type=float, default=0.7)
ap.add_argument("--max-det", type=int, default=1000)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--tile", type=int, default=1024)
ap.add_argument("--stride", type=int, default=768)
ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N dev images")
a = ap.parse_args()

W = a.workspace
sys.path.insert(0, str(W / "deps_ds"))
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor  # noqa: E402
import transformers  # noqa: E402

if Path(a.run_name).name != a.run_name:
    ap.error("run-name must be one directory name")
run = W / "runs_ds" / a.run_name
run.mkdir(parents=True, exist_ok=False)
t_start = time.perf_counter()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def save(name, obj):
    (run / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def log(s):
    line = time.strftime("%H:%M:%S") + " " + str(s)
    print(line, flush=True)
    with (run / "run.log").open("a", encoding="utf8") as f:
        f.write(line + "\n")


def fail(kind, value, tb):
    save("status.json", {"status": "failed", "error": str(value)})
    log("FAILED " + "".join(traceback.format_exception(kind, value, tb)))


sys.excepthook = fail
save("status.json", {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")})

old = W / "runs/vistas_sweep_v2_highres"
m = read(old / "manifest.json")
gt = read(old / "dev_coco.json")
ground_truth = read(old / "ground_truth.json")
dev = m["dev"][: a.limit] if a.limit else m["dev"]
cats = m["categories"]
plan = read(W / "reports/vistas_world_matrix_v4/plan.json")
phrases = plan["prompt_sets"][a.prompt_set] if a.prompt_set in plan["prompt_sets"] else m["prompts"]
assert len(phrases) == len(cats) == 13
root = W / "data/Vistas/v2_selected/training/images"
weights_dir = W / "weights/GDino_T_ds"

torch.manual_seed(17)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.6)

processor = GroundingDinoProcessor.from_pretrained(str(weights_dir))
model = GroundingDinoForObjectDetection.from_pretrained(str(weights_dir)).to("cuda").eval()
for p in model.parameters():
    p.requires_grad_(False)

# One text string holds all 13 phrases, separated by " . " so the tokenizer keeps
# them apart. Per-class score = max sigmoid logit over that phrase's own tokens,
# the standard Grounding DINO phrase score; it is not an average over the string.
text = " . ".join(phrases) + " ."
char_spans = []
cursor = 0
for ph in phrases:
    char_spans.append((cursor, cursor + len(ph)))
    cursor += len(ph) + 3
tok = processor.tokenizer
enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
sub_ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
inputs = processor(images=np.zeros((a.imgsz, a.imgsz, 3), dtype=np.uint8), text=text, return_tensors="pt")
ids = inputs["input_ids"][0].tolist()
j0 = next(
    (j for j in range(len(ids) - len(sub_ids) + 1) if ids[j : j + len(sub_ids)] == sub_ids),
    None,
)
if j0 is None:
    raise RuntimeError("cannot align tokenizer offsets with processor input_ids")
spans = []
for lo, hi in char_spans:
    idx = [j0 + k for k, (s, e) in enumerate(offsets) if s >= lo and e <= hi and e > s]
    if not idx:
        raise RuntimeError("empty token span for phrase")
    spans.append([idx[0], idx[-1] + 1])
log(f"phrases={phrases}")
log(f"token spans={spans} text_tokens={len(ids)}")

model_sha = hashlib.sha256((weights_dir / "model.safetensors").read_bytes()).hexdigest()
params = {
    "case": a.run_name,
    "weights_repo": "IDEA-Research/grounding-dino-tiny",
    "weights_sha256": model_sha,
    "model_parameter_count": sum(p.numel() for p in model.parameters()),
    "prompt_set": a.prompt_set,
    "prompts": phrases,
    "phrase_score": "max of sigmoid logits over the phrase's own tokens",
    "text_string": text,
    "token_spans": spans,
    "image_size": a.imgsz,
    "tile_original_pixels": a.tile,
    "stride": a.stride,
    "include_full_image": True,
    "per_view_max_detections": a.max_det,
    "final_max_detections": a.max_det,
    "confidence": a.conf,
    "nms_iou": a.nms,
    "batch_size": 1,
    "precision": "FP32",
    "cuda_memory_fraction": 0.6,
    "cpu_threads": 4,
    "seed": 17,
    "dev_images": len(dev),
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": torch.cuda.get_device_name(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "dev_list_sha256": hashlib.sha256(json.dumps(m["dev"]).encode()).hexdigest(),
    "ground_truth_sha256": hashlib.sha256((old / "dev_coco.json").read_bytes()).hexdigest(),
}
save("resolved_parameters.json", params)
save("source_split_manifest.json", m)
save("manifest.json", {**m, "run_type": "inference-only", "training_performed": False, "train": []})
save("ground_truth.json", ground_truth)
save("dev_coco.json", gt)
save("source_snapshot.json", {"source": Path(__file__).read_text(encoding="utf8")})


def infer_view(bgr):
    """Letterbox to imgsz, run Grounding DINO, return boxes in original pixels."""
    h, w = bgr.shape[:2]
    s = a.imgsz / max(h, w)
    nw, nh = round(w * s), round(h * s)
    left, top = (a.imgsz - nw) // 2, (a.imgsz - nh) // 2
    canvas = np.full((a.imgsz, a.imgsz, 3), 114, np.uint8)
    canvas[top : top + nh, left : left + nw] = cv2.resize(bgr, (nw, nh))
    inp = processor(images=canvas[:, :, ::-1], text=text, return_tensors="pt")
    inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.inference_mode():
        out = model(**inp)
    scores = out.logits[0].sigmoid()  # (queries, tokens)
    boxes = out.pred_boxes[0].float()  # cxcywh, normalized
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw / 2) * a.imgsz
    y1 = (cy - bh / 2) * a.imgsz
    x2 = (cx + bw / 2) * a.imgsz
    y2 = (cy + bh / 2) * a.imgsz
    rows = []
    for ci, (lo, hi) in enumerate(spans):
        sc = scores[:, lo:hi].max(dim=-1).values
        keep = sc >= a.conf
        if not bool(keep.any()):
            continue
        k = torch.nonzero(keep, as_tuple=True)[0]
        rows.append(
            torch.stack(
                [x1[k], y1[k], x2[k], y2[k], sc[k], torch.full_like(sc[k], float(ci))], dim=1
            )
        )
    if not rows:
        return np.zeros((0, 6), np.float32)
    out_arr = torch.cat(rows).cpu().numpy()
    out_arr[:, [0, 2]] = np.clip((out_arr[:, [0, 2]] - left) / (nw / w), 0, w)
    out_arr[:, [1, 3]] = np.clip((out_arr[:, [1, 3]] - top) / (nh / h), 0, h)
    return out_arr


def pos(n):
    return sorted(set(list(range(0, max(n - a.tile, 0) + 1, a.stride)) + [max(n - a.tile, 0)]))


torch.cuda.reset_peak_memory_stats()
log(f"START {a.run_name} {params['model_parameter_count']} parameters, dev={len(dev)}")
pred, timing, view_counts = [], [], []
start = time.perf_counter()
for idx, k in enumerate(dev):
    st = time.perf_counter()
    im = cv2.imread(str(root / (k + ".jpg")))
    if im is None:
        raise FileNotFoundError(root / (k + ".jpg"))
    pieces = [infer_view(im)]
    for top in pos(im.shape[0]):
        for left in pos(im.shape[1]):
            tile = infer_view(im[top : top + a.tile, left : left + a.tile])
            tile[:, [0, 2]] += left
            tile[:, [1, 3]] += top
            pieces.append(tile)
    arr = np.concatenate(pieces)
    arr = arr[(arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])]
    view_counts.append({"image": k, "views": len(pieces), "raw_boxes": int(len(arr))})
    inds = batched_nms(
        torch.from_numpy(arr[:, :4]), torch.from_numpy(arr[:, 4]), torch.from_numpy(arr[:, 5]), a.nms
    ).numpy()
    arr = arr[inds[: a.max_det]]
    for b in arr:
        pred.append(
            {
                "image_id": idx,
                "category_id": int(b[5]),
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "score": float(b[4]),
            }
        )
    timing.append(time.perf_counter() - st)
    elapsed = time.perf_counter() - start
    eta = elapsed / (idx + 1) * (len(dev) - idx - 1)
    save(
        "progress.json",
        {"done": idx + 1, "total": len(dev), "elapsed_seconds": elapsed, "eta_seconds": eta},
    )
    log(
        f"{idx+1}/{len(dev)} | views={len(pieces)} | boxes={len(arr)} | elapsed={elapsed:.1f}s "
        f"ETA={eta:.1f}s | CUDA={torch.cuda.memory_allocated()/2**20:.0f} MiB"
    )

save("predictions_local.json", pred)
save("view_counts.json", view_counts)

with contextlib.redirect_stdout(io.StringIO()):
    co = COCO()
    co.dataset = gt
    co.createIndex()
    dt = co.loadRes(pred)
    ev = COCOeval(co, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
pr = ev.eval["precision"]
per = {}
for i, c in enumerate(cats):
    v, v50 = pr[:, :, i, 0, 2], pr[0, :, i, 0, 2]
    per[c] = {
        "AP50": float(v50[v50 >= 0].mean()) if (v50 >= 0).any() else None,
        "AP": float(v[v >= 0].mean()) if (v >= 0).any() else None,
    }
summary = {
    "AP50": float(ev.stats[1]),
    "AP": float(ev.stats[0]),
    "seen_AP50": float(np.mean([per[c]["AP50"] for c in cats[:8]])),
    "novel_AP50": float(np.mean([per[c]["AP50"] for c in cats[8:]])),
    "classes": per,
    "median_end_to_end_seconds": float(np.median(timing[1:])),
    "inference_total_seconds": sum(timing),
    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    "parameter_count": params["model_parameter_count"],
    "predictions": len(pred),
}
save("summary.json", summary)
save("timings.json", timing)
log("RESULT " + json.dumps({k: v for k, v in summary.items() if k != "classes"}))
for c in cats:
    log(f"  {c}: AP50={per[c]['AP50']}")

try:
    from visualize_vistas_run import render_run

    render_run(
        W,
        old,
        out=run / "visual_samples",
        base_path=W / "reports/vistas_diagnostic_v3/tiled1000_predictions_local.json",
        tuned_path=run / "predictions_local.json",
        labels=("S tuned30 reference", a.run_name),
        caption="Fixed full image + tiles: " + a.run_name,
    )
except Exception as exc:  # visualisation must not invalidate a finished evaluation
    log("visualisation failed: " + repr(exc))

md = (
    "# "
    + a.run_name
    + "\n\n"
    + "|Parameter|Value|\n|---|---|\n"
    + "|Model|Grounding DINO-T (IDEA-Research/grounding-dino-tiny)|\n"
    + f"|Prompt set|{a.prompt_set}|\n"
    + "|Epoch|0 (inference only)|\n"
    + "|Batch|1|\n"
    + f"|Input|{a.imgsz} letterbox|\n"
    + f"|Original-pixel tile / stride|{a.tile} / {a.stride}|\n"
    + "|Precision|FP32|\n"
    + f"|Confidence / NMS / max detections|{a.conf} / {a.nms} / {a.max_det}|\n\n"
    + "## Results\n```json\n"
    + json.dumps(summary, indent=2)
    + "\n```\n\n## Full parameters\n```json\n"
    + json.dumps(params, indent=2)
    + "\n```\n\n![Fixed samples](visual_samples/fixed_samples.jpg)\n"
)
(run / "report.md").write_text(md, encoding="utf8")
save("status.json", {"status": "completed", "elapsed_seconds": time.perf_counter() - t_start})
log("COMPLETE")
