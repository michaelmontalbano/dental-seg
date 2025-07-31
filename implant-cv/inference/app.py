import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "dir/inference/pipeline/models/merged_implants.pt"
IMAGE_DIR = "dir/inference/demo_images"
GROUND_TRUTH_ID = 33  # Straumann ITI

# --- Load Model ---
model = YOLO(MODEL_PATH)
class_names = model.names

# --- Gather Images ---
images = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# --- Session State ---
if "index" not in st.session_state:
    st.session_state.index = 0

# --- Prediction Logic ---
def predict_image(img_path):
    results = model(img_path)[0]
    predictions = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = class_names[cls_id]
        predictions.append((cls_id, label, conf))
    return predictions

def next_image():
    if st.session_state.index < len(images) - 1:
        st.session_state.index += 1

# --- UI ---
st.title("Implant Classification Demo")
st.subheader(f"Image {st.session_state.index + 1} of {len(images)}")

# Get current image
img_path = images[st.session_state.index]
img_name = os.path.basename(img_path)

# Predict
if os.path.exists(img_path):
    st.image(img_path, caption=img_name, use_container_width=True)
    predictions = predict_image(img_path)

    if predictions:
        top_cls_id, top_label, top_conf = predictions[0]
        correct = top_cls_id == GROUND_TRUTH_ID
        result = "✅ Correct (Straumann)" if correct else f"❌ Incorrect: {top_label} ({top_conf:.2f})"
        st.markdown(f"### {result}")
    else:
        st.markdown("⚠️ No prediction.")
else:
    st.warning(f"⚠️ Image not found: {img_path}")

# --- Button ---
st.button("Next →", on_click=next_image)