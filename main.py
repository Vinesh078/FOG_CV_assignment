#!/usr/bin/env python3
"""
End-to-end pipeline: raw folder of images -> clustered identities with confidence scores.

Usage:
    python main.py --input data/sample_images --output output

Run `python main.py --help` for all options.
"""
import argparse
import time
from pathlib import Path

from src.extract import FaceExtractor, save_records
from src.cluster import cluster_faces
from src.report import write_outputs


def main():
    parser = argparse.ArgumentParser(description="Cluster images by identity using face embeddings.")
    parser.add_argument("--input", "-i", required=True, help="Folder of unorganized images")
    parser.add_argument("--output", "-o", default="output", help="Where to write results (default: output/)")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto-detect)")
    parser.add_argument("--cache-embeddings", action="store_true",
                         help="Save extracted embeddings to output/embeddings.json so re-clustering "
                              "(e.g. after changing eps) doesn't require re-running detection")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    extractor = FaceExtractor(device=args.device)
    records = extractor.run(input_dir)
    t1 = time.time()
    print(f"[main] detection + embedding took {t1 - t0:.1f}s ({(t1-t0)/max(len(records),1)*1000:.0f}ms/face)")

    if args.cache_embeddings:
        save_records(records, output_dir / "embeddings.json")

    if not records:
        print("[main] No faces found anywhere in the input directory. Nothing to cluster.")
        return

    results = cluster_faces(records)
    t2 = time.time()
    print(f"[main] clustering took {t2 - t1:.1f}s")

    write_outputs(results, input_dir, output_dir)
    print(f"[main] done in {time.time() - t0:.1f}s total. See {output_dir}/results.json and "
          f"{output_dir}/clusters/")


if __name__ == "__main__":
    main()
