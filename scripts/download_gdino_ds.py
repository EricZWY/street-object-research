"""Download the Grounding DINO-T checkpoint into weights/GDino_T_ds.

Kept separate from the YOLO-World weight downloader so the two families never
share a directory. Records bytes, sha256 and the resolved source URL so the run
can be reproduced after the cache is cleared.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

WS = Path(r"E:\CodexWorkspace\StreetObject_DataPlan")
REPO_ID = "IDEA-Research/grounding-dino-tiny"
DEST = WS / "weights/GDino_T_ds"

# The xet backend pre-allocates the 657 MiB shard and, when the connection stalls,
# leaves a sparse file whose header is all zeros; safetensors then refuses to load
# it. Plain HTTP is slower but sequential and self-verifying, so force it on.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

DEST.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(DEST),
    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.py"],
    max_workers=4,
)
print(f"snapshot at {path} in {time.perf_counter()-t0:.1f}s", flush=True)

files = []
for p in sorted(DEST.rglob("*")):
    if p.is_file() and p.name != "download_manifest.json":
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        files.append(
            {
                "path": str(p.relative_to(DEST)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": h.hexdigest(),
            }
        )
        print(f"{p.stat().st_size/2**20:8.1f} MiB  {p.relative_to(DEST)}", flush=True)

(DEST / "download_manifest.json").write_text(
    json.dumps(
        {
            "repo_id": REPO_ID,
            "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version,
            "files": files,
        },
        indent=2,
    ),
    encoding="utf8",
)
print("DONE", flush=True)
