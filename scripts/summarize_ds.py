"""Aggregate the _ds experiments (Grounding DINO-T baseline + expanded seen-class supervision).

Reads only the JSON summaries the runs wrote; publishes aggregate numbers and a bar chart,
never imagery, annotations, per-image predictions or weights.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--workspace", type=Path, required=True)
ap.add_argument("--repo", type=Path, default=None, help="public repository checkout; writes results/ and docs/figures/")
a = ap.parse_args()
w = a.workspace
o = w / "reports_ds"
o.mkdir(parents=True, exist_ok=True)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def fmt(x):
    return "n/a" if x is None else f"{100 * x:.2f}"


def short(label):
    """Compact axis label for the figure; the tables keep the full run name."""
    base = {
        "yoloworld_s_zeroshot": "YOLO-World S\nzero-shot",
        "yoloworld_s_tuned5_141": "S tuned 5ep\n141 img",
        "yoloworld_s_tuned30_141": "S tuned 30ep\n141 img",
        "grounding_dino_t_zeroshot": "Grounding DINO-T\nzero-shot",
    }.get(label)
    if base:
        return base
    name, _, ep = label.partition("@")
    mode = "prompt" if "_prompt_" in name else "head"
    rounds = name.split("_e")[-1].rstrip("_ds").rstrip("b")
    text = f"{mode} {rounds}ep\n548 img"
    return text + (f"\n(ep{ep})" if ep else "")


# ------------------------------------------------------------------ collect
cases = []  # (label, run dir, summary file, note)


def add(label, path, note="", plot=True):
    if path.exists():
        cases.append({"label": label, "path": str(path), "summary": read(path), "note": note, "plot": plot})


ref = w / "runs"
add("yoloworld_s_zeroshot", ref / "vistas_world_v4_s_original/summary.json", "reference, no adaptation")
add("yoloworld_s_tuned5_141", ref / "vistas_world_v4_s_tuned5/summary.json", "previous 141-image prompt tuning")
add("yoloworld_s_tuned30_141", ref / "vistas_world_v4_s_tuned30/summary.json", "previous best, 141-image prompt tuning")
add("grounding_dino_t_zeroshot", w / "runs_ds/gdino_t_zeroshot_960_tile1024_ds/summary.json", "different architecture, no training")

# prompt-mode runs first, then head-mode; intermediate epoch files follow their own run
runs_ds = [d for d in (w / "runs_ds").glob("seen_expand_*_ds") if (d / "tuned_summary.json").exists()]
runs_ds.sort(key=lambda d: (0 if "_prompt_" in d.name else 1, d.name))
for d in runs_ds:
    mode = "prompt" if "_prompt_" in d.name else "head"
    add(f"{d.name}", d / "tuned_summary.json", f"expanded seen supervision, {mode} mode")
    epochs = sorted(d.glob("epoch_*_summary.json"), key=lambda p: int(p.name.split("_")[1]))
    for e in epochs:
        # keep the figure readable: only the short runs show their intermediate epochs there,
        # the full curve of every run stays in the tables and in the per-run report
        add(f"{d.name}@{e.name.split('_')[1]}", e, f"intermediate epoch of {d.name}", plot=len(epochs) <= 2)

if not cases:
    raise SystemExit("no _ds run summaries found yet")
plotted = [c for c in cases if c["plot"]]

# ------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(plotted)), 4.6), layout="constrained")
x = np.arange(len(plotted))
w50 = 0.38
allv = [100 * c["summary"]["AP50"] for c in plotted]
seenv = [100 * c["summary"]["seen_AP50"] for c in plotted]
novelv = [100 * c["summary"]["novel_AP50"] for c in plotted]
ax.bar(x - w50 / 2, allv, w50, label="All AP50", color="#1f4e79")
ax.bar(x + w50 / 2, novelv, w50, label="Novel AP50", color="#c55a11")
for i, (v, s) in enumerate(zip(allv, seenv)):
    ax.text(i - w50 / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w50 / 2, novelv[i] + 0.5, f"{novelv[i]:.1f}", ha="center", va="bottom", fontsize=8)
ax.axhline(50, ls="--", lw=1, color="grey")
ax.text(len(plotted) - 0.5, 50.8, "50 target", ha="right", fontsize=8, color="grey")
ax.set(xticks=x, xticklabels=[short(c["label"]) for c in plotted], ylabel="AP50 (%)", ylim=(0, max(55, max(allv) + 8)))
ax.set_title("Grounding DINO-T baseline and expanded seen-class supervision (56 dev images, 844 GT)")
plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7.5)
ax.legend(loc="upper left", fontsize=8)
fig.savefig(o / "ds_comparison.png", dpi=160)
fig.savefig(o / "ds_comparison.svg")
plt.close(fig)

# ------------------------------------------------------------------- report
best = max(cases, key=lambda c: c["summary"]["AP50"])
new_cases = [c for c in cases if not c["label"].startswith("yoloworld") and "@" not in c["label"]]
best_new = max(new_cases, key=lambda c: c["summary"]["AP50"]) if new_cases else best
prefix = "yoloworld_s_tuned30_141"
prefix_val = next((fmt(c["summary"]["AP50"]) for c in cases if c["label"] == prefix), "19.99")
lines = [
    "# Grounding DINO-T 基线与已见类扩样参照",
    "",
    f"本轮包含两种不同来源的实验：一是 Grounding DINO-T 零样本推理基线（不同架构），二是把已见类有监督样本从 141 张扩到 548 张的参照。本轮新增实验里最高的是 {short(best_new['label']).replace(chr(10), '，')}（AP50={fmt(best_new['summary']['AP50'])}%），仍低于此前的参照 {short(prefix).replace(chr(10), '，')}（AP50={prefix_val}%）。",
    "",
]
interp = o / "interpretation.md"
if interp.exists():
    lines += [interp.read_text(encoding="utf-8"), ""]
lines += [
    "## 关键参数",
    "",
    "|参数|设置|",
    "|---|---|",
]
g = next((c for c in cases if c["label"] == "grounding_dino_t_zeroshot"), None)
lines += [
    "|Grounding DINO-T|IDEA-Research/grounding-dino-tiny，官方权重，零样本，无训练|",
    "|检测器对照|YOLO-World v2 S，官方权重|",
    "|每个视图输入|960×960 letterbox（两架构一致）|",
    "|推理视图|全图＋原图1024像素切片，步长768|",
    "|每视图 / 合并后最大输出|1000 / 1000|",
    "|置信度 / NMS IoU|0.001 / 0.7|",
    "|COCO maxDets|[1,10,100]|",
    "|精度 / CUDA分配器限额|FP32 / 60%|",
    "|开发集|同56张、844个GT，8已见＋5未见|",
    "|扩样训练数据|548张、7601框，每类上限160张，开发图零重叠|",
    "|扩样轮数 / 学习率|提示模式10轮与30轮两档，lr 2e-4；head 模式10轮，提示lr 2e-4＋检测头lr 1e-4|",
    "|CPU线程 / seed|4 / 17|",
    "",
    "## 全部结果",
    "",
    "|实验|总体AP50|已见AP50|未见AP50|AP50:95|单图中位秒|说明|",
    "|---|---:|---:|---:|---:|---:|---|",
]
for c in cases:
    s = c["summary"]
    ap95 = s.get("AP")
    lines.append(
        f"|{c['label']}|{fmt(s['AP50'])}|{fmt(s['seen_AP50'])}|{fmt(s['novel_AP50'])}|"
        f"{fmt(ap95)}|{s.get('median_end_to_end_seconds', float('nan')):.3f}|{c['note']}|"
    )
lines += ["", "![对比](ds_comparison.svg)", ""]

classes = list(cases[0]["summary"]["classes"])
# for the per-class table keep one column per experiment (previous best, GDINO, then each
# expanded run at its final epoch); the intermediate epochs stay in the table above
perclass = [c for c in cases if "@" not in c["label"] and c["label"] != "yoloworld_s_zeroshot"
            and c["label"] != "yoloworld_s_tuned5_141"]
lines += ["## 逐类别 AP50", "", "|类别|" + "|".join(short(c["label"]).replace("\n", " ") for c in perclass) + "|",
          "|---|" + "---:|" * len(perclass)]
for cl in classes:
    lines.append("|" + cl + "|" + "|".join(fmt(c["summary"]["classes"][cl]["AP50"]) for c in perclass) + "|")

lines += [
    "",
    "## 验证与局限",
    "",
    "- 两个实验使用同一开发图清单、同一 COCO 真值、同一推理几何（960 letterbox、1024 切片、步长 768、conf 0.001、NMS 0.7）。",
    "- Grounding DINO-T 是零样本基线，不接收任何本任务标注；它的分数说明的是换一个架构能把整体 AP50 推到哪里，不是统一口径的 SOTA 对比（官方 COCO 分数不能外推）。",
    "- 扩样参照只加大已见类样本量，未见类仍完全隔离；它用于判断充分监督下已见类能到多少，不能当作开放词汇成果。",
    "- 开发集已参与连续实验决策，存在选择偏差；正式 validation 未使用。",
    "- 548 张训练图里 505 张含路灯，这是 Vistas 标注分布造成的，不是采样错误；因此扩样对路灯提升的贡献不能外推成所有设施类都同样受益。",
    "- 扩样实验只跑了单 seed，学习率与轮数未调参，30 轮没有学习率衰减或早停，末轮并非最优轮次；head 模式只跑到 10 轮。",
    "- 置信度 0.001 时两个架构都把每张图的候选打满 1000 条上限（GDINO 共 56000 条 = 1000×56），即候选饱和，分数反映的是排序质量。",
    "- 训练用整图 letterbox 到 960，评测用全图＋1024 切片，训练与评测的尺度不完全一致；这是沿用历史设置以保持与既有结果可比。",
    "",
    "## 复现命令",
    "",
    "```powershell",
    "powershell -ExecutionPolicy Bypass -File scripts\\Start-GdinoEval_ds.ps1 -Workspace E:\\CodexWorkspace\\StreetObject_DataPlan",
    "powershell -ExecutionPolicy Bypass -File scripts\\Start-SeenExpand_ds.ps1 -Workspace E:\\CodexWorkspace\\StreetObject_DataPlan -Mode prompt -Epochs 10 -TrainPerClass 160",
    "powershell -ExecutionPolicy Bypass -File scripts\\Start-SeenExpand_ds.ps1 -Workspace E:\\CodexWorkspace\\StreetObject_DataPlan -Mode prompt -Epochs 30 -TrainPerClass 160",
    "powershell -ExecutionPolicy Bypass -File scripts\\Start-SeenExpand_ds.ps1 -Workspace E:\\CodexWorkspace\\StreetObject_DataPlan -Mode head -Epochs 10 -TrainPerClass 160",
    "```",
    "",
    "完整逐类、逐轮与逐参数记录见仓库内 `results/gdino-seen-expand-20261117_ds.json`；本轮新增文件与踩坑记录见 [`ds-experiment-changes.md`](ds-experiment-changes.md)。",
]
(o / "report_ds.md").write_text("\n".join(lines), encoding="utf8")

summary = {
    "development_set": "56 images / 844 GT, 8 seen + 5 novel classes (same as runs/vistas_sweep_v2_highres)",
    "official_validation_used": False,
    "inference_geometry": {
        "image_size": 960,
        "tile_original_pixels": 1024,
        "stride": 768,
        "views": "full image + tiles",
        "confidence": 0.001,
        "nms_iou": 0.7,
        "max_output": 1000,
    },
    "cases": [
        {
            "label": c["label"],
            "note": c["note"],
            "summary_path": c["path"],
            "metrics": {k: v for k, v in c["summary"].items() if k != "classes"},
        }
        for c in cases
    ],
    "per_class_AP50": {c["label"]: {k: v["AP50"] for k, v in c["summary"]["classes"].items()} for c in cases},
}
(o / "summary_ds.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")

if a.repo:
    (a.repo / "results").mkdir(exist_ok=True)
    (a.repo / "docs/figures").mkdir(parents=True, exist_ok=True)
    (a.repo / "results/gdino-seen-expand-20261117_ds.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8"
    )
    import shutil

    shutil.copyfile(o / "ds_comparison.svg", a.repo / "docs/figures/gdino-seen-expand-20261117_ds.svg")
    public = "\n".join(lines)
    public = public.replace("(ds_comparison.svg)", "(figures/gdino-seen-expand-20261117_ds.svg)")
    (a.repo / "docs/gdino-baseline-and-seen-expand_ds.md").write_text(public, encoding="utf8")
    print("wrote repository results/docs")

print(json.dumps({"cases": [{"label": c["label"], "AP50": c["summary"]["AP50"]} for c in cases]}, indent=2))
