"""Read-only probe: how large is the seen-class training pool at different caps?

Answers one question before committing to an expensive run: with the "all
annotated segments must be seen classes, no crowd" rule, how many distinct
images are available per seen class and as a union? Writes a small JSON so the
expansion size in the experiment document is auditable.
"""
import hashlib
import json
from pathlib import Path

WS = Path(r"E:\CodexWorkspace\StreetObject_DataPlan")
OUT = WS / "reports_ds" / "seen_expand_pool_probe.json"

cfg = json.loads((WS / "reports/vistas_preparation/class_split_v1.json").read_text(encoding="utf-8-sig"))
cats = cfg["seen"] + cfg["unseen"]
ns = len(cfg["seen"])
root = WS / "data/Vistas/v2_selected/training"
raw = json.loads((root / "v2.0/panoptic/panoptic_2020.json").read_text())
cid = {x["id"]: cats.index(x["name"]) for x in raw["categories"] if x["name"] in cats}
ann = {x["image_id"]: x for x in raw["annotations"]}
group = {k: [s for s in v["segments_info"] if s["category_id"] in cid] for k, v in ann.items()}


def order(keys, salt):
    return sorted(keys, key=lambda x: hashlib.sha256((salt + str(x)).encode()).hexdigest())


eligible = [k for k, ss in group.items() if ss and all(cid[s["category_id"]] < ns and not s["iscrowd"] for s in ss)]
per_class = {c: [k for k in eligible if any(cid[s["category_id"]] == c for s in group[k])] for c in range(ns)}
report = {
    "rule": "all annotated segments are seen classes, no crowd",
    "eligible_images": len(eligible),
    "per_class_available": {cats[c]: len(per_class[c]) for c in range(ns)},
    "union_by_cap": {},
}
for cap in [32, 64, 96, 128, 160, 200, 256, 320]:
    train = set()
    for c in range(ns):
        train.update(order(per_class[c], "train-prompt-v2")[:cap])
    report["union_by_cap"][cap] = len(train)
report["union_all"] = len(
    {k for c in range(ns) for k in order(per_class[c], "train-prompt-v2")[: len(per_class[c])]}
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
print(json.dumps(report, indent=2, ensure_ascii=False))
