"""
Stage 2: turn a pile of (face_id, embedding) pairs into named clusters, one per
identity, plus a per-image confidence score.

Why DBSCAN and not k-means: we don't know how many people are in the dataset ahead
of time, and k-means needs k. DBSCAN discovers the number of clusters from density
and, just as importantly, has a built-in concept of "noise" -- a face that doesn't
look like it belongs to any group gets labeled -1 instead of being forced into the
nearest cluster. For an identity system, "I'm not sure, here's a face I couldn't
confidently place" is a much better failure mode than silently misassigning someone.

Distance metric: cosine, not Euclidean. FaceNet-style embeddings are trained with a
loss that cares about direction, not magnitude, so cosine distance lines up with how
the embedding space was actually shaped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from . import config
from .extract import FaceRecord


@dataclass
class ClusterResult:
    cluster_id: str          # "person_01", "person_02", ... or "unmatched"
    face_id: str
    image_path: str
    bbox: list
    confidence: float        # 0-1, similarity to this cluster's centroid, rescaled


def cluster_faces(records: list[FaceRecord]) -> list[ClusterResult]:
    if not records:
        return []

    embeddings = np.array([r.embedding for r in records])
    # L2-normalize so cosine distance behaves and downstream centroid math is simple
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dist_matrix = cosine_distances(embeddings)

    db = DBSCAN(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES,
        metric="precomputed",
    )
    labels = db.fit_predict(dist_matrix)

    results: list[ClusterResult] = []

    unique_labels = sorted(set(labels) - {-1})
    label_to_name = {lbl: f"person_{i+1:02d}" for i, lbl in enumerate(unique_labels)}

    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        cluster_embeddings = embeddings[idx]
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        sims = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        # Rescale against a FIXED anchor tied to the DBSCAN eps, not the min/max of this
        # particular cluster. Rescaling per-cluster was the first thing I tried, and it's
        # wrong: in a 2-photo cluster the lower-similarity member always gets pinned to 0%
        # regardless of how good the actual match is, which misrepresents confidence for
        # exactly the small clusters where a reviewer most needs an honest number.
        # sim_floor is roughly "how similar to the centroid can a face be while still
        # being right at the DBSCAN decision boundary" -- (1 - eps), since cosine
        # similarity = 1 - cosine distance for L2-normalized vectors.
        sim_floor = 1 - config.DBSCAN_EPS
        sim_ceil = 1.0
        for i, sim in zip(idx, sims):
            conf = float(np.clip(
                (sim - sim_floor) / (sim_ceil - sim_floor),
                config.CONFIDENCE_FLOOR, config.CONFIDENCE_CEIL,
            ))
            r = records[i]
            results.append(
                ClusterResult(
                    cluster_id=label_to_name[lbl],
                    face_id=r.face_id,
                    image_path=r.image_path,
                    bbox=r.bbox,
                    confidence=round(conf, 4),
                )
            )

    # noise points: still reported, not silently dropped, with confidence 0 since
    # by definition DBSCAN couldn't tie them to a dense group with confidence
    for i in np.where(labels == -1)[0]:
        r = records[i]
        results.append(
            ClusterResult(
                cluster_id="unmatched",
                face_id=r.face_id,
                image_path=r.image_path,
                bbox=r.bbox,
                confidence=0.0,
            )
        )

    print(f"[cluster] {len(unique_labels)} identity cluster(s), "
          f"{sum(1 for l in labels if l == -1)} unmatched face(s)")
    return results
