# Evaluation: 3-person test set, realistic conditions

`data/eval_set/` — 6 photos, 3 people, 2 photos each (`person_0N_0.jpg` / `person_0N_1.jpg`
naming = ground truth). Unlike the clean portrait sanity check in `sanity_check.md`,
these are the kind of photos the assignment is actually going to be graded on: a bowling
alley at night, colored disco lighting, motion blur, and — importantly — other people
visible in the background of every shot. This is a much more honest test of whether the
pipeline works than four well-lit portraits.

Command:
```bash
python main.py --input data/eval_set --output output/eval_run
```

## Result

```
6 images -> 9 faces detected
4 clusters, 1 unmatched
```

Full output is checked in under `eval_run_output/`. The headline number: **all 3 ground-truth
identities were clustered correctly, with zero cross-contamination between them.**

| Ground truth | My cluster | Confidence |
|---|---|---|
| person_01 (bearded man, blue polo) | `person_01` | 76% / 76% |
| person_02 (woman, pink stripes) | `person_03` | 77% / 77% |
| person_03 (man, mustache, purple shirt) | `person_04` | 95% / 95% |

(Cluster *numbers* don't line up with the ground-truth filenames — DBSCAN doesn't know
the filenames, it just discovers groups and I name them in whatever order they're found.
What matters is each pair of photos of the same person landed in the same cluster, and
no photo of a different person joined it.)

## The two clusters I didn't expect, and why they're not actually errors

9 faces were detected across 6 photos, not 6 — every one of these photos has bystanders
in the background (it's a group bowling outing, not a studio shoot). Those extra faces
had to go *somewhere*, and this is what happened to them:

- **`person_02` cluster**: two background faces, from two *different* photos
  (`person_01_0.jpg` and `person_02_0.jpg`), clustered together at 76% confidence. I
  checked the crops — it looks like the same bystander happens to appear in both shots,
  which makes sense: it's the same friend group at the same event, so background people
  recur across photos of different foreground subjects. I can't fully verify this is the
  same person vs. a coincidental lookalike from a crop this small, but it's a plausible,
  defensible clustering decision either way — not an obvious mistake.
- **`unmatched`**: one small, blurry background face (roughly 22x28px in the original
  image) that didn't confidently match anything, correctly landed in `unmatched` instead
  of getting forced into a cluster.

I'd call both of these the pipeline doing the right thing with information the ground
truth labels don't cover, rather than a failure — but I'm flagging them explicitly
instead of quietly excluding them, because a clustering system that silently drops
"inconvenient" faces to make its numbers look cleaner is a worse thing to hand in than
one that shows its actual output.

## What this test does and doesn't tell me

**Does tell me:** the pipeline holds up on real-world lighting and motion blur, correctly
separates 3 people with zero confusion between them, and degrades sensibly (lower
confidence, or `unmatched`) on the hard/small/incidental faces rather than failing
silently or crashing.

**Doesn't tell me:** how it performs at real dataset scale, on lookalikes/family members,
or on faces at more extreme angles than these photos happen to contain. Confidence is
also visibly lower here (76-95%) than on the clean portraits (82-99%) in the other sanity
check, which is exactly what I'd expect given the lighting/blur — but with only 3 people
I can't yet say where the confidence number stops being trustworthy as a ranking signal.
That's the kind of question a larger labeled set would answer and this one can't.
