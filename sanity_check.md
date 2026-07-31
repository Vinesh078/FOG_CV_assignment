# Sanity check: does the pipeline actually cluster correctly?

Before trusting this against the real assignment dataset, I ran it against 4
public-domain photos as a known-answer check: two named individuals ("Person A" and
"Person B" below — public figures, kept generic here since the point is the pipeline,
not who's in the photo), 2 solo photos of Person A, 1 solo photo of Person B, and 1
group photo containing both of them together (so the pipeline also has to handle
multiple faces in a single image correctly).

Command:
```bash
python main.py --input data/sample_images --output output
```

Output:
```
[extract] 4 images -> 5 faces detected
[main] detection + embedding took 1.4s (280ms/face)
[cluster] 2 identity cluster(s), 0 unmatched face(s)
[main] clustering took 0.0s
[report] wrote output/results.json and crops under output/clusters
```

`output/results.json` (abbreviated):

```json
{
  "num_clusters": 2,
  "num_faces": 5,
  "num_unmatched": 0,
  "clusters": {
    "person_01": [
      {"image_path": "biden.jpg",       "confidence": 0.9936},
      {"image_path": "two_people.jpg",  "confidence": 0.9936}
    ],
    "person_02": [
      {"image_path": "obama.jpg",       "confidence": 0.9155},
      {"image_path": "obama2.jpg",      "confidence": 0.8676},
      {"image_path": "two_people.jpg",  "confidence": 0.8237}
    ]
  }
}
```

## What this confirms

- **Both faces in the group photo were correctly split** into the right existing
  clusters rather than merged with each other or dropped — this was the part I was
  least sure would work out of the box.
- **Multi-image identity grouping works**: Person A's two separate solo photos, taken
  at different times/angles, landed in the same cluster.
- **Confidence tracks something real, not just cluster membership**: the group-photo
  faces (smaller, angled, partially side-lit) score lower than the solo portraits, which
  matches what I'd expect a harder crop to do to embedding similarity.

## What this does NOT confirm

Four photos, two identities, no adversarial cases (no lookalikes, no twins, no heavy
makeup/aging gap, no low-light or motion blur) is not a benchmark — it's a check that
the pipeline isn't obviously broken before pointing it at a real dataset. I'm not
claiming an accuracy number from this; see the README's "Limitations" section for what
I'd actually want before making an accuracy claim.
