"""
Stage 3: write results in two forms —

1. output/clusters/person_01/, person_02/, ... unmatched/ — actual image crops
   (with the confidence score burned into the filename) so a human reviewer can
   open a folder and *look* at whether the clustering is right. A JSON file alone
   is not how anyone actually QAs a vision system.
2. output/results.json — machine-readable, matches the assignment's ask for
   "confidence/accuracy score assigned to each image within a cluster".
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from .cluster import ClusterResult


def write_outputs(results: list[ClusterResult], input_dir: Path, output_dir: Path):
    output_dir = Path(output_dir)
    clusters_dir = output_dir / "clusters"
    if clusters_dir.exists():
        shutil.rmtree(clusters_dir)
    clusters_dir.mkdir(parents=True)

    by_cluster = defaultdict(list)
    for r in results:
        by_cluster[r.cluster_id].append(r)

    for cluster_id, items in by_cluster.items():
        cdir = clusters_dir / cluster_id
        cdir.mkdir(parents=True, exist_ok=True)
        for item in sorted(items, key=lambda x: -x.confidence):
            src = Path(input_dir) / item.image_path
            try:
                img = Image.open(src)
                img = ImageOps.exif_transpose(img).convert("RGB")
                x1, y1, x2, y2 = [int(v) for v in item.bbox]
                # pad the crop a bit so the face isn't uncomfortably tight-cropped
                w, h = img.size
                pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
                crop = img.crop((
                    max(0, x1 - pad_x), max(0, y1 - pad_y),
                    min(w, x2 + pad_x), min(h, y2 + pad_y),
                ))
                conf_pct = int(round(item.confidence * 100))
                out_name = f"{conf_pct:03d}pct__{item.face_id}.jpg"
                crop.save(cdir / out_name, quality=90)
            except Exception as e:
                print(f"[report] could not write crop for {item.face_id}: {e}")

    report = {
        "num_clusters": len([c for c in by_cluster if c != "unmatched"]),
        "num_faces": len(results),
        "num_unmatched": len(by_cluster.get("unmatched", [])),
        "clusters": {
            cluster_id: [
                {
                    "face_id": r.face_id,
                    "image_path": r.image_path,
                    "bbox": r.bbox,
                    "confidence": r.confidence,
                }
                for r in sorted(items, key=lambda x: -x.confidence)
            ]
            for cluster_id, items in by_cluster.items()
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[report] wrote {output_dir/'results.json'} and crops under {clusters_dir}")
    return report
