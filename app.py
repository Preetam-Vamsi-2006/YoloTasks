import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Video Object Detection",
    page_icon="🎥",
    layout="wide"
)

# -------------------------------------------------
# Custom UI Styling
# -------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00ffcc;
        text-align: center;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🚀 YOLOv8 Video Object Detection")
st.write("Upload a video and detect objects using **YOLOv8**")

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.3, 0.05
)

# -------------------------------------------------
# Load YOLO Model (Cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
if "stop" not in st.session_state:
    st.session_state.stop = False

if "processing" not in st.session_state:
    st.session_state.processing = False

# -------------------------------------------------
# Video Upload
# -------------------------------------------------
uploaded_video = st.file_uploader(
    "📤 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

# -------------------------------------------------
# Stop Button
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Processing"):
        st.session_state.stop = False
        st.session_state.processing = True

with col2:
    if st.button("⛔ Stop Processing"):
        st.session_state.stop = True
        st.session_state.processing = False

# -------------------------------------------------
# Video Processing Logic
# -------------------------------------------------
if uploaded_video is not None and st.session_state.processing:

    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(tempfile.gettempdir(), "yolo_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()
    progress_bar = st.progress(0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():

        # Stop condition
        if st.session_state.stop:
            st.warning("🛑 Processing stopped by user")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference (REFERENCE FROM YOUR ORIGINAL CODE)
        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        # Display in Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(
            annotated_frame,
            channels="RGB",
            use_column_width=True
        )

        current_frame += 1
        progress_bar.progress(current_frame / frame_count)

    cap.release()
    out.release()

    st.session_state.processing = False

    st.success("✅ Processing completed")

    # -------------------------------------------------
    # Download Button
    # -------------------------------------------------
    with open(output_path, "rb") as f:
        st.download_button(
            "⬇️ Download Processed Video",
            data=f,
            file_name="yolo_output.mp4",
            mime="video/mp4"
        )
