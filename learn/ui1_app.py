"""
Bai 12: UI — giao dien tim kiem video tren trinh duyet.

Dung Streamlit: viet Python thuan → ra web app.
Khong can HTML/CSS/JavaScript.

CACH CHAY:
  conda activate aic
  streamlit run learn/ui1_app.py
"""
import streamlit as st
import torch
import open_clip
import faiss
import numpy as np
import json
import os
import glob
from PIL import Image

# =================================================================
# CAU HINH TRANG
# =================================================================
st.set_page_config(
    page_title="AIC 2026 — Video Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AIC 2026 — Video Search")
st.caption("Go mo ta bang tieng Anh → tim keyframe khop nhat")

# =================================================================
# TAI MODEL (chi tai 1 lan, cache lai)
# =================================================================
@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    return model, preprocess, tokenizer

model, preprocess, tokenizer = load_clip()

# =================================================================
# ENCODE KEYFRAME (chi chay 1 lan, cache lai)
# =================================================================
@st.cache_data
def load_keyframes_and_encode():
    keyframe_dir = "data/keyframes"
    paths = sorted(glob.glob(f"{keyframe_dir}/**/*.jpg", recursive=True))

    if not paths:
        return [], None

    all_features = []
    batch_size = 32

    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            img = Image.open(p).convert("RGB")
            batch.append(preprocess(img))

        batch_tensor = torch.stack(batch)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
        all_features.append(features)

    features = torch.cat(all_features, dim=0)
    features = features / features.norm(dim=1, keepdim=True)
    features_np = features.numpy().astype(np.float32)

    # Tao FAISS index
    index = faiss.IndexFlatIP(features_np.shape[1])
    index.add(features_np)

    return paths, index

image_paths, index = load_keyframes_and_encode()

# =================================================================
# TAI OCR RESULTS (neu co)
# =================================================================
ocr_data = {}
ocr_path = "data/ocr_results.json"
if os.path.exists(ocr_path):
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

# =================================================================
# TAI ASR RESULTS (neu co)
# =================================================================
asr_data = {}
asr_path = "data/asr_results.json"
if os.path.exists(asr_path):
    with open(asr_path, "r", encoding="utf-8") as f:
        asr_data = json.load(f)

# =================================================================
# TAI YOLO DETECT RESULTS (neu co)
# =================================================================
detect_data = {}
detect_path = "data/detect_results.json"
if os.path.exists(detect_path):
    with open(detect_path, "r", encoding="utf-8") as f:
        detect_data = json.load(f)

# =================================================================
# GIAO DIEN TIM KIEM
# =================================================================
if not image_paths:
    st.error("Khong tim thay keyframe. Chay keyframe1_extract.py truoc.")
    st.stop()

# Thanh tim kiem
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("🔍 Mo ta canh can tim:", placeholder="a cat playing with yarn")
with col2:
    search_mode = st.selectbox("Che do:", [
        "CLIP (hinh anh)", "OCR (tim chu)", "ASR (giong noi)",
        "YOLO (vat the)", "Temporal (chuoi hanh dong)"
    ])

k = st.slider("So ket qua:", min_value=3, max_value=20, value=5)

# =================================================================
# XU LY TIM KIEM
# =================================================================
if query:
    if search_mode == "CLIP (hinh anh)":
        # CLIP search
        text_tokens = tokenizer([query])
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_np = text_features.numpy().astype(np.float32)

        scores, ids = index.search(text_np, k)

        st.markdown(f"### Ket qua CLIP cho: *\"{query}\"*")

        # Hien thi anh theo grid
        cols = st.columns(min(5, k))
        for rank in range(k):
            idx = ids[0][rank]
            score = scores[0][rank]
            path = image_paths[idx]
            name = os.path.basename(path)

            with cols[rank % 5]:
                st.image(path, caption=f"#{rank+1} | {score:.3f}\n{name}", use_container_width=True)

    elif search_mode == "OCR (tim chu)":
        # OCR search
        if not ocr_data:
            st.warning("Chua co du lieu OCR. Chay ocr1_demo.py truoc.")
        else:
            results = []
            query_lower = query.lower()
            for name, texts in ocr_data.items():
                for t in texts:
                    if query_lower in t.lower():
                        results.append((name, t))

            st.markdown(f"### Ket qua OCR cho: *\"{query}\"*")

            if not results:
                st.info(f"Khong tim thay chu \"{query}\" trong keyframe nao.")
            else:
                st.success(f"Tim thay {len(results)} ket qua!")
                cols = st.columns(min(5, len(results)))
                for i, (name, text) in enumerate(results[:k]):
                    full_path = None
                    for p in image_paths:
                        if os.path.basename(p) == name:
                            full_path = p
                            break

                    with cols[i % 5]:
                        if full_path:
                            st.image(full_path, caption=f"{name}\nOCR: \"{text}\"", use_container_width=True)
                        else:
                            st.write(f"{name}: \"{text}\"")

    elif search_mode == "ASR (giong noi)":
        # ASR search
        if not asr_data:
            st.warning("Chua co du lieu ASR. Chay asr1_demo.py truoc.")
        else:
            results = []
            query_lower = query.lower()
            for video_name, data in asr_data.items():
                for seg in data.get("segments", []):
                    if query_lower in seg["text"].lower():
                        results.append({
                            "video": video_name,
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"]
                        })

            st.markdown(f"### Ket qua ASR cho: *\"{query}\"*")

            if not results:
                st.info(f"Khong tim thay \"{query}\" trong giong noi.")
            else:
                st.success(f"Tim thay {len(results)} doan noi!")
                for r in results[:k]:
                    # Tim keyframe gan timestamp nhat
                    target_time = r["start"]
                    best_path = None
                    best_diff = float("inf")
                    for p in image_paths:
                        name = os.path.basename(p)
                        # Parse timestamp tu ten file: test_001740_58.0s.jpg
                        parts = name.replace(".jpg", "").split("_")
                        for part in parts:
                            if part.endswith("s"):
                                try:
                                    t = float(part[:-1])
                                    diff = abs(t - target_time)
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_path = p
                                except ValueError:
                                    pass

                    col_img, col_text = st.columns([1, 2])
                    with col_img:
                        if best_path:
                            st.image(best_path, use_container_width=True)
                    with col_text:
                        st.write(f"**[{r['start']:.1f}s - {r['end']:.1f}s]**")
                        st.write(f"*\"{r['text']}\"*")
                        st.write(f"Video: {r['video']}")

    elif search_mode == "YOLO (vat the)":
        # YOLO search — tim frame co vat the cu the
        # Query format: "person" hoac "2 person" hoac "person, car"
        if not detect_data:
            st.warning("Chua co du lieu YOLO. Chay yolo1_detect.py truoc.")
        else:
            st.markdown(f"### Ket qua YOLO cho: *\"{query}\"*")
            st.caption("VD: `person` | `2 person` | `person, motorcycle`")

            # Parse query: "2 person, 1 car" → {person: 2, car: 1}
            # Hoac don gian: "person" → {person: 1}
            requirements = {}
            parts = [p.strip() for p in query.split(",")]
            for part in parts:
                tokens = part.strip().split()
                if len(tokens) >= 2 and tokens[0].isdigit():
                    count = int(tokens[0])
                    obj_name = " ".join(tokens[1:])
                    requirements[obj_name] = count
                elif len(tokens) >= 1:
                    obj_name = " ".join(tokens)
                    requirements[obj_name] = 1

            results = []
            for name, objects in detect_data.items():
                match = True
                for req_obj, req_count in requirements.items():
                    found = objects.get(req_obj, 0)
                    if found < req_count:
                        match = False
                        break
                if match:
                    total = sum(objects.values())
                    results.append((name, objects, total))

            results.sort(key=lambda x: -x[2])

            if not results:
                st.info(f"Khong tim thay frame nao khop.")
                all_objects = set()
                for objects in detect_data.values():
                    all_objects.update(objects.keys())
                st.write("Cac vat the co san:", ", ".join(sorted(all_objects)))
            else:
                st.success(f"Tim thay {len(results)} frame khop!")
                cols = st.columns(min(5, len(results[:k])))
                for i, (name, objects, total) in enumerate(results[:k]):
                    full_path = None
                    for p in image_paths:
                        if os.path.basename(p) == name:
                            full_path = p
                            break

                    obj_str = ", ".join(f"{v} {k}" for k, v in objects.items())
                    with cols[i % 5]:
                        if full_path:
                            st.image(full_path, caption=f"{name}\n{obj_str}", use_container_width=True)
                        else:
                            st.write(f"{name}: {obj_str}")

    elif search_mode == "Temporal (chuoi hanh dong)":
        # Temporal search — tim chuoi hanh dong theo thu tu thoi gian
        st.markdown(f"### Temporal Search")
        st.caption("Nhap cac buoc cach nhau boi ` → ` hoac ` then `. VD: `busy street → rice field → food market`")

        # Parse steps
        if " → " in query:
            steps = [s.strip() for s in query.split(" → ")]
        elif " then " in query.lower():
            steps = [s.strip() for s in query.lower().split(" then ")]
        else:
            steps = [query.strip()]

        if len(steps) < 2:
            st.warning("Can it nhat 2 buoc. VD: `street scene → temple → sunset`")
        else:
            st.info(f"Tim chuoi {len(steps)} buoc: {' → '.join(steps)}")

            # Parse timestamp tu ten file
            def get_timestamp(path):
                name = os.path.basename(path)
                parts = name.replace(".jpg", "").split("_")
                for part in parts:
                    if part.endswith("s"):
                        try:
                            return float(part[:-1])
                        except ValueError:
                            pass
                return -1.0

            ts_list = [get_timestamp(p) for p in image_paths]

            # CLIP search cho tung buoc
            top_per_step = 20
            step_results = []
            for step_query in steps:
                text_tokens = tokenizer([step_query])
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                text_np = text_features.numpy().astype(np.float32)
                scores, ids = index.search(text_np, top_per_step)
                hits = [(ids[0][j], scores[0][j]) for j in range(top_per_step)]
                step_results.append(hits)

            # Greedy: chon best chain thoa man thoi gian tang dan
            best_chain = []
            best_total_score = -1

            for start_idx, start_score in step_results[0][:10]:
                chain = [(start_idx, start_score)]
                current_time = ts_list[start_idx]

                for step_i in range(1, len(steps)):
                    best_next = None
                    best_next_score = -1
                    for idx, score in step_results[step_i]:
                        t = ts_list[idx]
                        if t > current_time:
                            if score > best_next_score:
                                best_next = (idx, score)
                                best_next_score = score
                    if best_next is None:
                        break
                    chain.append(best_next)
                    current_time = ts_list[best_next[0]]

                if len(chain) == len(steps):
                    total_score = sum(s for _, s in chain)
                    if total_score > best_total_score:
                        best_total_score = total_score
                        best_chain = chain

            if not best_chain:
                st.error("Khong tim thay chuoi frame hop le theo thu tu thoi gian.")
            else:
                st.success(f"Tim thay chuoi! Tong score = {best_total_score:.3f}")
                cols = st.columns(len(best_chain))
                for i, (idx, score) in enumerate(best_chain):
                    path = image_paths[idx]
                    name = os.path.basename(path)
                    t = ts_list[idx]
                    with cols[i]:
                        st.image(path, use_container_width=True)
                        st.caption(f"Buoc {i+1}: {steps[i]}\n{name} (t={t:.1f}s, score={score:.3f})")

# =================================================================
# THONG TIN HE THONG
# =================================================================
with st.sidebar:
    st.markdown("## 📊 Thong tin")
    st.write(f"**Keyframe:** {len(image_paths)}")
    st.write(f"**OCR data:** {len(ocr_data)} frame co chu")
    st.write(f"**ASR data:** {len(asr_data)} video da nghe")
    st.write(f"**YOLO data:** {len(detect_data)} frame co vat the")
    st.write(f"**Model:** ViT-B-32 (OpenAI)")
    st.markdown("---")
    st.markdown("### 💡 Meo")
    st.markdown("""
    - **CLIP**: mo ta canh (tieng Anh tot hon)
    - **OCR**: tim chu cu the trong anh
    - **ASR**: tim theo loi noi trong video
    - **YOLO**: tim vat the cu the (so luong)
    - **Temporal**: tim chuoi hanh dong
    - VD CLIP: `a man riding bicycle`
    - VD OCR: `Toyota` hoac `EXIT`
    - VD ASR: `hello` hoac `xin chao`
    - VD YOLO: `person` hoac `2 person, car`
    - VD Temporal: `street → temple → sunset`
    """)
