"""
Central config so nobody has to go hunting through the codebase to tune a threshold.

Everything here has a comment on WHY the value is what it is, not just what it is,
because "0.62 seemed to work on my laptop" is not a defensible answer in an interview.
"""

# --- Detection ---
MIN_FACE_SIZE = 40          # px; anything smaller is almost always a false positive or unusable crop
DETECTION_CONF_THRESHOLD = 0.90   # MTCNN's own objectness score, not identity confidence

# --- Embedding ---
EMBEDDING_MODEL = "vggface2"  # facenet-pytorch pretrained option; "casia-webface" is the alternative
EMBEDDING_DIM = 512

# --- Clustering ---
# DBSCAN on cosine distance. eps is the max distance for two faces to be considered
# "close enough" to link. This is the single most important number in the whole project.
#
# Empirically (see notebooks/threshold_tuning.ipynb) FaceNet embeddings of the SAME person
# under normal lighting/pose variation sit at cosine distance ~0.25-0.55. Different people
# are usually >0.75. 0.55 is a deliberately slightly-conservative choice: it costs a bit of
# recall (occasionally splits one person into two clusters under extreme lighting) in
# exchange for not merging two different people, which is the worse failure mode for an
# identity system.
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 2      # a "cluster" of one photo isn't a cluster, it's a singleton -> noise/unmatched

# --- Confidence scoring ---
# Confidence per image = cosine similarity of that face's embedding to its cluster's
# centroid, min-max rescaled so 1.0 = dead center of the cluster, 0.0 = right at the
# eps boundary. This is intentionally NOT a calibrated probability (that would need a
# labeled validation set to fit a proper sigmoid/Platt scaling), and the README says so.
CONFIDENCE_FLOOR = 0.0
CONFIDENCE_CEIL = 1.0

# --- Misc ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42
