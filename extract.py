"""
Stage 1: walk the input directory, detect every face in every image, and turn each
face into a 512-d embedding.

One image can contain zero, one, or several faces -- a photo from a party is not
guaranteed to have exactly one person in it. So the unit of work downstream isn't
"image", it's "face": each detected face becomes its own record with a pointer back
to the source image and its bounding box, because two people in the same photo need
to end up in two different clusters, not get lumped in as one blurred identity.

Detector: MTCNN (Zhang et al., 2016) via facenet-pytorch -- multi-stage cascade, gives
bounding boxes, landmarks, and a detection-probability score for free, and it's fast
enough on CPU for a dataset in the hundreds-to-low-thousands range.

Embedder: InceptionResnetV1 pretrained on VGGFace2 (facenet-pytorch). 512-d embedding
space trained with a triplet/softmax objective so that L2/cosine distance between
embeddings of the same identity is small and different identities is large. This is
the same family of model (FaceNet) that made the "cluster faces by embedding distance"
approach standard in the first place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1

from . import config


@dataclass
class FaceRecord:
    face_id: str            # unique id: "<image_stem>__f<index>"
    image_path: str         # relative path to the source image
    bbox: list               # [x1, y1, x2, y2] in original image pixel coords
    det_score: float        # MTCNN's detection confidence, 0-1
    embedding: list          # 512-d list (kept as list for clean JSON serialization)


class FaceExtractor:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # keep_all=True: we want every face in the frame, not just the most prominent one
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=config.MIN_FACE_SIZE,
            post_process=False,
        )
        self.embedder = InceptionResnetV1(pretrained=config.EMBEDDING_MODEL).eval().to(self.device)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path)
        # phones/cameras write EXIF orientation flags instead of rotating pixels;
        # ignoring this silently rotates a chunk of any real-world dataset 90/180 degrees
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    def process_image(self, path: Path, rel_path: str) -> list[FaceRecord]:
        img = self._load_image(path)

        boxes, probs = self.detector.detect(img)
        if boxes is None:
            return []

        records: list[FaceRecord] = []
        crops = self.detector.extract(img, boxes, save_path=None)
        if crops is None:
            return []
        if crops.ndim == 3:  # single face -> add batch dim
            crops = crops.unsqueeze(0)

        with torch.no_grad():
            crops = crops.to(self.device)
            # facenet-pytorch's fixed_image_standardization, applied manually since
            # post_process=False above (we want the raw crop for confidence displays too)
            normed = (crops - 127.5) / 128.0
            embeddings = self.embedder(normed).cpu().numpy()

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < config.DETECTION_CONF_THRESHOLD:
                continue
            records.append(
                FaceRecord(
                    face_id=f"{Path(rel_path).stem}__f{i}",
                    image_path=rel_path,
                    bbox=[float(x) for x in box],
                    det_score=float(prob),
                    embedding=embeddings[i].tolist(),
                )
            )
        return records

    def run(self, input_dir: Path) -> list[FaceRecord]:
        input_dir = Path(input_dir)
        paths = sorted(
            p for p in input_dir.rglob("*")
            if p.suffix.lower() in config.SUPPORTED_EXTS
        )
        if not paths:
            raise FileNotFoundError(
                f"No images with extensions {config.SUPPORTED_EXTS} found under {input_dir}"
            )

        all_records: list[FaceRecord] = []
        skipped = []
        for p in paths:
            rel = str(p.relative_to(input_dir))
            try:
                recs = self.process_image(p, rel)
                if not recs:
                    skipped.append(rel)
                all_records.extend(recs)
            except Exception as e:  # a single corrupt/truncated image shouldn't kill the whole run
                skipped.append(f"{rel} (error: {e})")

        if skipped:
            print(f"[extract] {len(skipped)}/{len(paths)} images produced no usable face "
                  f"(no face detected, or below confidence threshold). See extract.log for the list.")
            log_path = Path("output") / "extract_skipped.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(skipped))

        print(f"[extract] {len(paths)} images -> {len(all_records)} faces detected")
        return all_records


def save_records(records: list[FaceRecord], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in records], f)


def load_records(in_path: Path) -> list[FaceRecord]:
    with open(in_path) as f:
        data = json.load(f)
    return [FaceRecord(**d) for d in data]
