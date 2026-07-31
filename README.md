# Face Identity Clustering

Groups an unsorted folder of photos by the person in them — across different lighting,
angles, and expressions — and gives each photo a confidence score for its assigned
identity. Built for the FOG Computer Vision Engineer take-home (Q1: person
identification/clustering).

## Why this approach

The task is unconstrained face clustering with an unknown number of identities, so the
pipeline has to answer three separate questions, and I kept them as three separate
stages rather than one black box, mostly so each one is independently debuggable:

| Stage | What | Model / method | Why |
|---|---|---|---|
| Detect | Find every face in every photo | MTCNN (`facenet-pytorch`) | Multi-stage cascade with landmarks + a detection confidence score included for free; one photo can have 0, 1, or many faces, so the unit of work is "a face," not "an image" |
| Embed | Turn a face crop into a vector | InceptionResnetV1, pretrained on VGGFace2 | FaceNet-family model — trained so that embedding distance tracks identity, which is the whole premise clustering relies on |
| Cluster | Group vectors by identity | DBSCAN, cosine distance | Doesn't require knowing the number of people ahead of time (unlike k-means), and has a real concept of "noise" — a face that doesn't clearly belong anywhere becomes `unmatched` instead of getting forced into the nearest group |

Confidence per photo = cosine similarity of that face's embedding to its cluster's
centroid, rescaled against a fixed anchor tied to the clustering threshold (see
`src/cluster.py` for the exact math and, more importantly, the comment explaining why
the first, more obvious version of this — rescaling per-cluster — was wrong).

**This is a similarity score, not a calibrated probability.** Turning it into a real
probability ("87% confident") would need a labeled validation set to fit something like
Platt scaling, which the assignment's dataset doesn't provide. I'd rather report an
honest, well-defined similarity score than a probability-shaped number I can't back up.

## Results

Two checks, in increasing order of how much they actually prove:

**1. Clean-portrait sanity check** (`notebooks/sanity_check.md`) — 4 well-lit
public-domain photos, 2 people, one group shot containing both. This was mostly to
confirm the pipeline wasn't obviously broken before testing it on anything harder.
5 faces, 2 clusters, 0 unmatched, 82-99% confidence.

**2. Realistic-conditions test** (`notebooks/eval_set_results.md`) — 6 photos, 3 people,
2 photos each, shot at a bowling alley at night: colored disco lighting, motion blur,
and other people visible in the background of every shot. This is a much closer proxy
for what the actual assignment dataset probably looks like than four studio portraits.

```
6 images -> 9 faces detected (3 target subjects + incidental background people)
All 3 target identities clustered correctly, zero cross-contamination
Confidence: 76-95% (lower than the clean set, as expected given lighting/blur)
1 low-quality background face correctly landed in "unmatched" instead of being forced
```

Full crops and reasoning (including what I made of the extra clusters from background
faces, and what this test does *not* prove) are in `notebooks/eval_set_results.md` —
worth reading before assuming this generalizes, since 3 people is still a small sample.

**I did not have the actual assignment `person_identification` dataset while building
this** — both checks above are stand-ins. Swap in the real dataset and re-run before
submitting; see "Running on your dataset" below.

## Setup

```bash
pip install -r requirements.txt
```

No GPU required — MTCNN + InceptionResnetV1 both run fine on CPU for datasets in the
hundreds-to-low-thousands range. Pretrained weights (~110MB) download automatically on
first run.

## Usage

```bash
# Batch pipeline (this is the actual deliverable)
python main.py --input data/sample_images --output output

# Interactive demo (what I recorded the demo video against)
streamlit run app.py
```

Batch output:
- `output/results.json` — every face, its cluster assignment, bounding box, and confidence
- `output/clusters/person_01/`, `person_02/`, ... `unmatched/` — actual cropped face
  images, confidence burned into the filename, so you can eyeball correctness without
  parsing JSON

### Running on your dataset

```bash
python main.py --input /path/to/person_identification --output output --cache-embeddings
```

`--cache-embeddings` saves the extracted embeddings to `output/embeddings.json` so if
you want to retune `DBSCAN_EPS` in `src/config.py` afterward, you're not re-running
face detection from scratch — I added this after re-running the full pipeline a few
times myself to sanity-check the threshold.

## Project structure

```
main.py              CLI entry point (detect -> embed -> cluster -> write results)
app.py                Streamlit demo UI
src/
  config.py           All tunable thresholds, with reasoning for each default
  extract.py           Stage 1: face detection + embedding
  cluster.py            Stage 2: DBSCAN clustering + confidence scoring
  report.py             Stage 3: writes results.json + cropped image folders
data/sample_images/   4 clean-portrait sanity-check images
data/eval_set/         6 realistic-conditions test images (3 people, 2 photos each)
notebooks/             sanity_check.md, eval_set_results.md, and eval_run_output/
                        (checked-in copy of the eval_set run, so results are visible
                        without re-running the pipeline)
output/                generated by main.py — not checked in
```

