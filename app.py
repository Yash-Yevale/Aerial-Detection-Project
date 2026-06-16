import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO


# PAGE CONFIG

st.set_page_config(
    page_title="Aerial Object Detection",
    page_icon="🛰️",
    layout="wide"
)

# CSS Part

st.markdown("""
<style>

/* REMOVE STREAMLIT HEADER COMPLETELY */
header {visibility: hidden;}
[data-testid="stHeader"] {display: none;}
[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
.block-container {padding-top: 0rem !important;}

/* BACKGROUND */
.stApp {
    background-color: #0b0b0b;
}

/* TEXT */
h1, h2, h3, h4, h5, h6, p, span {
    color: white !important;
}

/* CONTROL PANEL */
.control-panel {
    background: #111;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #222;
    height: 100%;
}

/* CARD */
.card {
    background: #161616;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #222;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #f5f5f5;
    border-radius: 10px;
    padding: 15px;
}

[data-testid="stFileUploader"] * {
    color: black !important;
}

/* BUTTON */
[data-testid="stFileUploader"] button {
    background-color: black !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}

/* PROGRESS */
.stProgress > div > div > div {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# HEADER

st.markdown("<h1 style='text-align:center;'>🛰️ Aerial Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa;'>Bird 🐦 & Drone 🚁 Recognition</p>", unsafe_allow_html=True)

st.markdown("---")

# LAYOUT (CUSTOM CONTROL PANEL)

left, right = st.columns([1, 3])

# CONTROL PANEL (LEFT SIDE)

with left:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Control Panel")

    mode = st.radio(
        "Mode",
        ["📊 Classification", "📦 Detection"]
    )

    confidence_threshold = st.slider(
        "Detection Confidence",
        0.1, 1.0, 0.5, 0.05
    )

    st.markdown('</div>', unsafe_allow_html=True)

# LOAD MODELS

@st.cache_resource
def load_models():
    clf = tf.keras.models.load_model('final_model.h5')
    yolo = YOLO("runs/detect/train4/weights/best.pt")
    return clf, yolo

clf_model, yolo_model = load_models()
IMG_SIZE = (224, 224)

# MAIN CONTENT (RIGHT SIDE)

with right:

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # INPUT IMAGE
        with col1:
            st.markdown('<div class="card"><b>📷 Input Image</b></div>', unsafe_allow_html=True)
            st.image(image, width=400)

        # CLASSIFICATION
        if "Classification" in mode:
            img = image.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = clf_model.predict(img_array)

            if prediction[0][0] > 0.5:
                label = "🚁 Drone"
                confidence = float(prediction[0][0])
            else:
                label = "🐦 Bird"
                confidence = float(1 - prediction[0][0])

            with col2:
                st.markdown('<div class="card"><b>🧠 Prediction</b></div>', unsafe_allow_html=True)
                st.markdown(f"### {label}")
                st.write(f"Confidence: {confidence:.2f}")
                st.progress(confidence)

        # DETECTION
        else:
            img_array = np.array(image)
            results = yolo_model(img_array, conf=confidence_threshold)
            result_img = results[0].plot()

            with col2:
                st.markdown('<div class="card"><b>📦 Detection Output</b></div>', unsafe_allow_html=True)
                st.image(result_img, width=400)

                boxes = results[0].boxes
                if boxes is not None:
                    st.markdown("### 🔍 Detected Objects")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = "🐦 Bird" if cls == 0 else "🚁 Drone"
                        st.write(f"{label} — {conf:.2f}")

    else:
        st.markdown("""
        <div class="card" style="
            height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <p style="font-size:18px;">
                👆 Upload an image to start
            </p>
        </div>
        """, unsafe_allow_html=True)

