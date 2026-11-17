"""Seen-class supervision reference on the frozen Vistas development split.

Question this run answers: with the current Vistas panoptic labels and the
current class definition, does *more* supervision on the seen classes move the
seen AP50 substantially, or is the ceiling set by the data/annotation itself?

Two adapters over the same frozen YOLO-World S checkpoint:

  mode=prompt  only the 8 seen text embeddings are trained (4096 numbers), as in
               the small-data prompt-tuning runs, but on a much larger seen-class
               image pool.
  mode=head    the seen embeddings *and* the detection head are trained - a
               deliberately stronger "well supervised" reference. Novel-class
               embeddings stay frozen and are still reported, but they are
               expected to degrade; that is the point of the contrast.

The 56 development images, the 844 boxes and the tiling geometry are identical
to runs/vistas_sweep_v2_highres, so AP50 is directly comparable with the
19.10 / 19.99 prompt-tuning results.
"""
import argparse
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms
from ultralytics import YOLOWorld
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("--workspace", type=Path, required=True)
ap.add_argument("--run-name", required=True)
ap.add_argument("--mode", choices=["prompt", "head"], default="prompt")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=0.0002)
ap.add_argument("--head-lr", type=float, default=0.0001)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--train-per-class", type=int, default=160)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--tile", type=int, default=1024)
ap.add_argument("--stride", type=int, default=768)
ap.add_argument("--prepare-only", action="store_true", help="build the split, record it, then stop")
ap.add_argument("--reuse-split", type=Path, default=None, help="reuse manifest.json + ground_truth.json from an earlier prepare run instead of decoding panoptic PNGs again")
ap.add_argument("--smoke-steps", type=int, default=0, help="pipeline check: only N steps per epoch")
a = ap.parse_args()

if a.epochs < 1 or a.lr <= 0 or a.head_lr <= 0 or a.train_per_class < 1 or a.eval_every < 1:
    ap.error("invalid settings")
if Path(a.run_name).name != a.run_name:
    ap.error("run-name must be one directory name")

W = a.workspace
run = W / "runs_ds" / a.run_name
run.mkdir(parents=True, exist_ok=False)
t_start = time.perf_counter()
phase_starts = {}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def save(name, obj):
    (run / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def log(s):
    line = datetime.now().astimezone().isoformat(timespec="seconds") + " " + str(s)
    print(line, flush=True)
    with (run / "run.log").open("a", encoding="utf8") as f:
        f.write(line + "\n")


def progress(phase, done, total, detail=""):
    now = time.perf_counter()
    start = phase_starts.setdefault(phase, now)
    elapsed = now - start
    eta = elapsed / done * (total - done) if done >= 3 and elapsed > 0 else None
    save(
        "progress.json",
        {
            "phase": phase,
            "done": done,
            "total": total,
            "percent": 100 * done / total,
            "phase_elapsed_seconds": elapsed,
            "run_elapsed_seconds": now - t_start,
            "phase_eta_seconds": eta,
        },
    )
    est = f"{eta:.0f}s" if eta is not None else "estimating"
    log(f"{phase} {done}/{total} ({100*done/total:.1f}%) | elapsed={elapsed:.1f}s | ETA={est} | {detail}")


def failed(kind, value, tb):
    save("status.json", {"status": "failed", "error": str(value)})
    log("FAILED " + "".join(traceback.format_exception(kind, value, tb)))


sys.excepthook = failed
save("status.json", {"status": "running", "started": datetime.now().astimezone().isoformat()})

# ---------------------------------------------------------------- data selection
cfg = read(W / "reports/vistas_preparation/class_split_v1.json")
cats = cfg["seen"] + cfg["unseen"]
ns = len(cfg["seen"])
prompts = [
    "street light fixture",
    "electrical junction box",
    "trash can",
    "fire hydrant",
    "manhole cover",
    "storm drain",
    "utility pole",
    "store sign",
    "surveillance camera",
    "bench",
    "mailbox",
    "bicycle rack",
    "information sign",
]
assert len(prompts) == len(cats)
root = W / "data/Vistas/v2_selected/training"
raw = json.loads((root / "v2.0/panoptic/panoptic_2020.json").read_text())
cid = {x["id"]: cats.index(x["name"]) for x in raw["categories"] if x["name"] in cats}
imeta = {x["id"]: x for x in raw["images"]}
ann = {x["image_id"]: x for x in raw["annotations"]}
group = {k: [s for s in v["segments_info"] if s["category_id"] in cid] for k, v in ann.items()}


def order(keys, salt):
    return sorted(keys, key=lambda x: hashlib.sha256((salt + str(x)).encode()).hexdigest())


dev = read(W / "runs/vistas_sweep_v2_highres/manifest.json")["dev"]
assert len(dev) == 56

if a.reuse_split is not None:
    # Reuse an earlier prepare-only run: the split and the decoded boxes are identical
    # for the same --train-per-class, so decoding 604 panoptic PNGs again is pure cost.
    reuse = read(a.reuse_split / "manifest.json")
    train = list(reuse["train"])
    dev = list(reuse["dev"])
    records = read(a.reuse_split / "ground_truth.json")
    reuse_cap = int(re.search(r"\d+", reuse["train_selection"]).group(0))
    assert int(a.train_per_class) == reuse_cap, (
        f"--train-per-class {a.train_per_class} does not match the reused split (cap {reuse_cap})"
    )
    assert not (set(train) & set(dev))
    log(f"reused split from {a.reuse_split} | train={len(train)} dev={len(dev)}")
    phase_starts["prepare"] = time.perf_counter()
    progress("prepare", len(train) + len(dev), len(train) + len(dev), "reused decoded split")
else:
    dev_hash = {hashlib.sha256((root / "images" / imeta[k]["file_name"]).read_bytes()).hexdigest() for k in dev}
    eligible = [
        k
        for k, ss in group.items()
        if k not in dev
        and ss
        and all(cid[s["category_id"]] < ns and not s["iscrowd"] for s in ss)
    ]
    train = set()
    for c in range(ns):
        pool = [k for k in eligible if any(cid[s["category_id"]] == c for s in group[k])]
        for k in order(pool, "train-prompt-v2"):
            if hashlib.sha256((root / "images" / imeta[k]["file_name"]).read_bytes()).hexdigest() in dev_hash:
                continue
            train.add(k)
            if sum(1 for x in train if any(cid[s["category_id"]] == c for s in group[x])) >= a.train_per_class:
                break
    train = sorted(train)
    assert not (set(train) & set(dev))

    records = {}
    phase_starts["prepare"] = time.perf_counter()
    for j, k in enumerate(train + dev):
        img = cv2.imread(str(root / "v2.0/panoptic" / ann[k]["file_name"]))
        rgb = img[:, :, ::-1].astype(np.int32)
        ids = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]
        rr = []
        for s in group[k]:
            yy, xx = np.where(ids == s["id"])
            if len(xx) == 0:
                continue
            assert len(xx) == s["area"]
            rr.append({"class": cid[s["category_id"]], "box": [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]})
        records[k] = rr
        if (j + 1) % 50 == 0 or j + 1 == len(train) + len(dev):
            progress("prepare", j + 1, len(train) + len(dev), f"images decoded")

manifest = {
    "train": train,
    "dev": dev,
    "categories": cats,
    "prompts": prompts,
    "mode": a.mode,
    "epochs": a.epochs,
    "lr": a.lr,
    "head_lr": a.head_lr if a.mode == "head" else None,
    "seed": 17,
    "image_size": a.imgsz,
    "batch": 1,
    "precision": "FP32",
    "train_selection": (
        f"{a.train_per_class} images per seen class (SHA256 order, union), all annotated segments are seen "
        "classes, no crowd, excludes the fixed development images and byte-identical duplicates"
    ),
    "dev_selection": "the same 56 development images as runs/vistas_sweep_v2_highres",
    "official_validation_used": False,
}
save("manifest.json", manifest)
save("ground_truth.json", records)
log(f"train={len(train)} dev={len(dev)} mode={a.mode} per_class_cap={a.train_per_class}")
if a.prepare_only:
    save("status.json", {"status": "prepared_only"})
    log("PREPARE ONLY, stopping before model load")
    sys.exit(0)

# --------------------------------------------------------------------- model
torch.manual_seed(17)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.6)
weight = W / "weights/YOLOWorld/yolov8s-worldv2.pt"
yolo = YOLOWorld(str(weight))
yolo.set_classes(prompts)
m = yolo.model
m.clip_model = None
m.to("cuda").eval()
for p in m.parameters():
    p.requires_grad_(False)

initial = m.txt_feats.detach().clone().to("cuda").float()
learn = torch.nn.Parameter(initial[:, :ns].clone())
novel = initial[:, ns:].clone()
head = m.model[-1]
head.nc = ns
m.args = get_cfg()
criterion = v8DetectionLoss(m)

# The detector stays in eval() in both modes: Detect.forward then returns
# (inference_tensor, raw_feature_maps) and v8DetectionLoss consumes element [1],
# while BatchNorm statistics of the frozen trunk are left untouched.
head_params = []
head_snapshot = None
if a.mode == "head":
    for _, p in m.model[-1].named_parameters():
        p.requires_grad_(True)
        head_params.append(p)
    assert head_params, "detect head exposed no parameters"
    head_snapshot = {n: p.detach().clone() for n, p in m.model[-1].named_parameters()}
    log(f"head trainable parameters: {sum(p.numel() for p in head_params)}")

groups = [{"params": [learn], "lr": a.lr}]
if head_params:
    groups.append({"params": head_params, "lr": a.head_lr})
optimizer = torch.optim.AdamW(groups, weight_decay=0)

params_doc = {
    "manifest": manifest,
    "optimizer": "AdamW",
    "optimizer_defaults": optimizer.defaults,
    "param_groups": [{"lr": g["lr"], "names": ["seen_text_embeddings"] if i == 0 else ["detect_head"]} for i, g in enumerate(groups)],
    "loss_gains": {k: getattr(m.args, k) for k in ["box", "cls", "dfl"]},
    "scheduler": None,
    "warmup_steps": 0,
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 10,
    "augmentation": "none; OpenCV linear resize, centered pad 114, BGR->RGB, /255",
    "num_workers": 0,
    "cpu_threads": 4,
    "cuda_memory_fraction": 0.6,
    "trainable_parameters": int(learn.numel() + sum(p.numel() for p in head_params)),
    "frozen_detector_eval_mode": a.mode == "prompt",
    "checkpoint_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python": sys.version,
    "platform": platform.platform(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "ultralytics": ultralytics.__version__,
    "evaluation": {"imgsz": a.imgsz, "conf": 0.001, "nms_iou": 0.7, "max_det": 1000, "tile": a.tile, "stride": a.stride},
    "checkpoint_selection": "last epoch; periodic development metrics are diagnostic, no early stopping",
}
save("resolved_parameters.json", params_doc)

table = "| Parameter | Value |\n|---|---|\n" + "".join(
    f"| {k} | {v} |\n"
    for k, v in {
        "mode": a.mode,
        "epochs": a.epochs,
        "train images": len(train),
        "images per seen class (cap)": a.train_per_class,
        "development images": len(dev),
        "prompt lr / head lr": f"{a.lr} / {a.head_lr if a.mode=='head' else '-'}",
        "input size": a.imgsz,
        "batch": 1,
        "precision": "FP32",
        "trainable parameters": params_doc["trainable_parameters"],
        "seed": 17,
    }.items()
)
(run / "report.md").write_text(
    f"# {a.run_name}\n\nStatus: running.\n\n## Key parameters\n\n{table}\n"
    "## Detailed resolved parameters\n\n```json\n" + json.dumps(params_doc, indent=2) + "\n```\n",
    encoding="utf8",
)


def get(k):
    img = cv2.imread(str(root / "images" / imeta[k]["file_name"]))
    return get_tensor(img)


def infer_view(x):
    raw = m.predict(x, txt_feats=feats)[0]
    out = non_max_suppression(raw, conf_thres=0.001, iou_thres=0.7, max_det=1000, nc=len(cats))[0].cpu().numpy()
    return out


def pos(n):
    return sorted(set(list(range(0, max(n - a.tile, 0) + 1, a.stride)) + [max(n - a.tile, 0)]))


def evaluate(tag):
    """Tiled evaluation, identical geometry to the YOLO-World matrix."""
    global feats
    phase_starts[tag] = time.perf_counter()
    m.eval()
    head.nc = len(cats)
    # Evaluation needs all 13 text columns (8 seen + 5 novel): the head's class channel
    # count follows the text matrix, so it must match head.nc here, unlike training.
    with torch.no_grad():
        feats = torch.cat([torch.nn.functional.normalize(learn.detach(), dim=-1), novel], dim=1)
    gt = {"images": [], "categories": [{"id": i, "name": c} for i, c in enumerate(cats)], "annotations": []}
    pred, times = [], []
    aid = 1
    with torch.inference_mode():
        for idx, k in enumerate(dev):
            st = time.perf_counter()
            im = cv2.imread(str(root / "images" / imeta[k]["file_name"]))
            pieces = []
            for top in pos(im.shape[0]):
                for left in pos(im.shape[1]):
                    crop = im[top : top + a.tile, left : left + a.tile]
                    t, (sx, sy, l, tp) = get_tensor(crop)
                    out = infer_view(t)
                    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - l) / sx, 0, crop.shape[1]) + left
                    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - tp) / sy, 0, crop.shape[0]) + top
                    pieces.append(out)
            arr = np.concatenate(pieces)
            arr = arr[(arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])]
            keep = batched_nms(
                torch.from_numpy(arr[:, :4]), torch.from_numpy(arr[:, 4]), torch.from_numpy(arr[:, 5]), 0.7
            ).numpy()
            arr = arr[keep[:1000]]
            for b in arr:
                pred.append(
                    {
                        "image_id": idx,
                        "category_id": int(b[5]),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(b[4]),
                    }
                )
            times.append(time.perf_counter() - st)
            gt["images"].append({"id": idx, "width": imeta[k]["width"], "height": imeta[k]["height"]})
            for r in records[k]:
                x1, y1, x2, y2 = r["box"]
                gt["annotations"].append(
                    {
                        "id": aid,
                        "image_id": idx,
                        "category_id": r["class"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
                aid += 1
            progress(tag, idx + 1, len(dev), f"views={len(pieces)} boxes={len(arr)}")
    save(tag + "_predictions.json", pred)
    save("dev_coco.json", gt)
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
            "AP": float(v[v >= 0].mean()) if (v >= 0).any() else None,
            "AP50": float(v50[v50 >= 0].mean()) if (v50 >= 0).any() else None,
        }
    summary = {
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "seen_AP50": float(np.mean([per[c]["AP50"] for c in cats[:ns]])),
        "novel_AP50": float(np.mean([per[c]["AP50"] for c in cats[ns:]])),
        "seen_AP": float(np.mean([per[c]["AP"] for c in cats[:ns]])),
        "novel_AP": float(np.mean([per[c]["AP"] for c in cats[ns:]])),
        "classes": per,
        "median_end_to_end_seconds": float(np.median(times[1:])),
        "predictions": len(pred),
    }
    save(tag + "_summary.json", summary)
    log(f"{tag}: " + json.dumps({k: v for k, v in summary.items() if k != "classes"}))
    m.eval()
    return summary


def get_tensor(crop):
    h, w = crop.shape[:2]
    s = a.imgsz / max(h, w)
    nw, nh = round(w * s), round(h * s)
    left, top = (a.imgsz - nw) // 2, (a.imgsz - nh) // 2
    canvas = np.full((a.imgsz, a.imgsz, 3), 114, np.uint8)
    canvas[top : top + nh, left : left + nw] = cv2.resize(crop, (nw, nh))
    x = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1)[None].to("cuda").float() / 255
    return x, (nw / w, nh / h, left, top)


feats = torch.nn.functional.normalize(learn.detach(), dim=-1)
torch.cuda.reset_peak_memory_stats()
log("START " + a.run_name)
lossrows = []
phase_starts["training"] = time.perf_counter()
for ep in range(a.epochs):
    head.nc = ns
    rng = np.random.default_rng(17 + ep)
    order_ = rng.permutation(len(train))
    steps = order_[: a.smoke_steps] if a.smoke_steps else order_
    tot = []
    for step, j in enumerate(steps):
        k = train[int(j)]
        img = cv2.imread(str(root / "images" / imeta[k]["file_name"]))
        x, (sx, sy, l, t) = get_tensor(img)
        boxes, labels = [], []
        for r in records[k]:
            assert r["class"] < ns
            x1, y1, x2, y2 = r["box"]
            boxes.append([((x1 + x2) * sx / 2 + l) / a.imgsz, ((y1 + y2) * sy / 2 + t) / a.imgsz, (x2 - x1) * sx / a.imgsz, (y2 - y1) * sy / a.imgsz])
            labels.append(r["class"])
        batch = {
            "batch_idx": torch.zeros(len(labels), device="cuda"),
            "cls": torch.tensor(labels, device="cuda").view(-1, 1),
            "bboxes": torch.tensor(boxes, device="cuda").view(-1, 4),
        }
        optimizer.zero_grad()
        # Training uses seen-only text columns (ns rows) so that the WorldDetect head,
        # whose class channels come from the contrastive head over the text matrix,
        # matches head.nc = ns. Novel columns are only concatenated for evaluation.
        feats = torch.nn.functional.normalize(learn, dim=-1)
        out = m.predict(x, txt_feats=feats)
        loss, parts = criterion(out, batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert learn.grad is not None and torch.isfinite(learn.grad).all()
        if head_params:
            assert any(p.grad is not None for p in head_params)
        torch.nn.utils.clip_grad_norm_([learn] + head_params, 10)
        optimizer.step()
        tot.append(float(loss.detach()))
        lossrows.append({"epoch": ep + 1, "step": step + 1, "loss": tot[-1]})
        progress(
            "training",
            ep * len(steps) + step + 1,
            a.epochs * len(steps),
            f"epoch={ep+1}/{a.epochs} step={step+1}/{len(steps)} loss={tot[-1]:.4f} cuda={torch.cuda.memory_allocated()/2**20:.0f}MiB",
        )
    log(f"epoch {ep+1} mean loss {np.mean(tot):.4f}")
    ckpt = {"seen_embeddings": learn.detach().cpu(), "epoch": ep + 1}
    if head_params:
        ckpt["head_state"] = {n: p.detach().cpu() for n, p in m.model[-1].state_dict().items()}
    torch.save(ckpt, run / f"ckpt_epoch_{ep+1}.pt")
    save("loss.json", lossrows)
    if (ep + 1) % a.eval_every == 0 and ep + 1 < a.epochs:
        evaluate(f"epoch_{ep+1}")

final = torch.cat([torch.nn.functional.normalize(learn.detach(), dim=-1), novel], dim=1)
feats = final
assert torch.equal(final[:, ns:], initial[:, ns:])
tuned = evaluate("tuned")
torch.save({"text_embeddings": final.cpu(), "prompts": prompts, "mode": a.mode}, run / "tuned_prompts.pt")

audit = {
    "mode": a.mode,
    "trainable_parameters": params_doc["trainable_parameters"],
    "seen_embedding_l2_change": float((final[:, :ns] - initial[:, :ns]).norm()),
    "novel_embeddings_bitwise_unchanged": bool(torch.equal(final[:, ns:], initial[:, ns:])),
    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
}
if head_params:
    audit["head_param_l2_change"] = {
        n: float((p.detach() - head_snapshot[n]).norm()) for n, p in m.model[-1].named_parameters()
    }
    audit["head_changed_parameters"] = int(
        sum(1 for n, p in m.model[-1].named_parameters() if not torch.equal(p.detach(), head_snapshot[n]))
    )
save("audit.json", audit)
log("COMPLETE " + json.dumps(audit))

try:
    from visualize_vistas_run import render_run

    render_run(
        W,
        run,
        out=run / "visual_samples",
        base_path=W / "reports/vistas_diagnostic_v3/tiled1000_predictions_local.json",
        tuned_path=run / "tuned_predictions.json",
        labels=("S tuned30 reference", a.run_name),
        caption="Fixed full image + tiles: " + a.run_name,
    )
except Exception as exc:
    log("visualisation failed: " + repr(exc))

(run / "report.md").write_text(
    f"# {a.run_name}\n\nStatus: completed.\n\n## Key parameters\n\n{table}\n"
    "## Results\n\n```json\n" + json.dumps({"tuned": tuned, "audit": audit}, indent=2) + "\n```\n"
    "## Detailed resolved parameters\n\n```json\n" + json.dumps(params_doc, indent=2) + "\n```\n"
    "## Fixed visual samples\n\nThreshold 0.1 for display only.\n\n![Fixed samples](visual_samples/fixed_samples.jpg)\n",
    encoding="utf8",
)
save("status.json", {"status": "completed", "elapsed_seconds": time.perf_counter() - t_start})
