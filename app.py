"""
Streamlit demo UI. This is what I recorded the demo video against.

Not the "real" deliverable (that's main.py + src/, meant to run as a batch job over
a dataset) -- this exists so a reviewer, or me on a screen recording, can drag a folder
of photos in and actually see the clustering happen instead of reading a JSON file.
"""
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from src.extract import FaceExtractor
from src.cluster import cluster_faces

st.set_page_config(page_title="Face Identity Clustering", page_icon="\U0001F464", layout="wide")

st.title("Face Identity Clustering")
st.caption(
    "Upload a pile of unsorted photos. The system detects every face, embeds it with a "
    "FaceNet-style model, and groups faces that belong to the same person -- with a "
    "confidence score per photo, not just a hard yes/no."
)

with st.sidebar:
    st.header("How this works")
    st.markdown(
        "1. **Detect** every face in every photo (MTCNN)\n"
        "2. **Embed** each face into a 512-d vector (InceptionResnetV1 / VGGFace2)\n"
        "3. **Cluster** by cosine distance (DBSCAN — no need to know the number of "
        "people ahead of time)\n"
        "4. **Score** each photo by how close it sits to its cluster's centroid"
    )
    st.divider()
    st.markdown(
        "Faces the system isn't confident about land in **unmatched** rather than "
        "being forced into a cluster."
    )

uploaded = st.file_uploader(
    "Upload images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True
)

run = st.button("Cluster faces", type="primary", disabled=not uploaded)

if "extractor" not in st.session_state:
    with st.spinner("Loading models (first run only)..."):
        st.session_state.extractor = FaceExtractor()

if run and uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for f in uploaded:
            (tmpdir / f.name).write_bytes(f.getbuffer())

        progress = st.progress(0.0, text="Detecting faces...")
        records = []
        for i, f in enumerate(uploaded):
            recs = st.session_state.extractor.process_image(tmpdir / f.name, f.name)
            records.extend(recs)
            progress.progress((i + 1) / len(uploaded), text=f"Processed {f.name}")
        progress.empty()

        if not records:
            st.warning("No faces detected in any uploaded image.")
            st.stop()

        with st.spinner("Clustering..."):
            results = cluster_faces(records)

        by_cluster = {}
        for r in results:
            by_cluster.setdefault(r.cluster_id, []).append(r)

        n_people = len([c for c in by_cluster if c != "unmatched"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Faces detected", len(results))
        c2.metric("People identified", n_people)
        c3.metric("Unmatched faces", len(by_cluster.get("unmatched", [])))

        st.divider()

        # deterministic order, unmatched last
        ordered = sorted((c for c in by_cluster if c != "unmatched")) + (
            ["unmatched"] if "unmatched" in by_cluster else []
        )

        for cluster_id in ordered:
            items = sorted(by_cluster[cluster_id], key=lambda x: -x.confidence)
            label = "Unmatched faces" if cluster_id == "unmatched" else cluster_id.replace("_", " ").title()
            st.subheader(f"{label}  ({len(items)} photo{'s' if len(items) != 1 else ''})")

            cols = st.columns(min(len(items), 6) or 1)
            for i, item in enumerate(items):
                img = Image.open(tmpdir / item.image_path)
                img = ImageOps.exif_transpose(img).convert("RGB")
                x1, y1, x2, y2 = [int(v) for v in item.bbox]
                w, h = img.size
                pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
                crop = img.crop((
                    max(0, x1 - pad_x), max(0, y1 - pad_y),
                    min(w, x2 + pad_x), min(h, y2 + pad_y),
                ))
                with cols[i % len(cols)]:
                    st.image(crop, use_container_width=True)
                    if cluster_id == "unmatched":
                        st.caption(f"{item.image_path}")
                    else:
                        st.caption(f"{item.image_path} — **{item.confidence*100:.0f}%**")
elif not uploaded:
    st.info("Upload a set of photos (ideally with a few different people, some appearing "
            "more than once) to see the clustering.")
