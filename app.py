"""
Blind Assistant - AI-Powered Vision & Navigation
Streamlit web app with real-time YOLO object detection and Google Maps navigation.
"""
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import threading
import time
from collections import Counter
from ultralytics import YOLO
from navigator import get_walking_directions

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Blind Assistant | AI Vision & Navigation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(102,126,234,0.08));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    background: linear-gradient(135deg, #00d4aa, #667eea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem;
}
.main-header p { color: #8892b0; font-size: 1rem; margin: 0; }

.det-item {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.6rem 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;
    background: rgba(0,212,170,0.06); border-left: 3px solid #00d4aa;
    animation: slideIn 0.3s ease;
}
.det-item.warn { background: rgba(255,217,61,0.06); border-left-color: #ffd93d; }
.det-item.danger { background: rgba(255,107,107,0.06); border-left-color: #ff6b6b; }

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}

.nav-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(0,212,170,0.05));
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0;
}
.nav-active { border-color: #00d4aa; box-shadow: 0 0 15px rgba(0,212,170,0.3); }

.metric-box {
    text-align: center; padding: 1rem;
    background: rgba(255,255,255,0.03); border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}
.metric-box .value { font-size: 1.8rem; font-weight: 700; color: #00d4aa; }
.metric-box .label { font-size: 0.8rem; color: #8892b0; margin-top: 0.2rem; }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0e1225 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00b894) !important;
    color: #000 !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0,212,170,0.4) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL (cached across reruns)
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# Object categories for smart alerts
VEHICLE_OBJECTS = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane"}
WARNING_OBJECTS = {"fire hydrant", "stop sign", "traffic light", "dog"}

# ==========================================
# RTC CONFIGURATION
# ==========================================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# SESSION STATE
# ==========================================
for key, default in {
    "nav_steps": [], "nav_current": 0, "nav_active": False,
    "nav_summary": {}, "last_spoken": "", "speak_queue": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================
# VIDEO PROCESSOR (runs in separate thread)
# ==========================================
class BlindAssistantProcessor(VideoProcessorBase):
    confidence = 0.40

    def __init__(self):
        self._lock = threading.Lock()
        self.detections = []
        self.detection_counts = {}
        self.stability = {}
        self.STABLE_FRAMES = 3

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=self.confidence, verbose=False)[0]

        detected = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detected.append(label)

            # Color by category
            if label in VEHICLE_OBJECTS:
                color = (61, 217, 255)  # yellow-ish BGR
            elif label in WARNING_OBJECTS:
                color = (107, 107, 255)  # red-ish BGR
            else:
                color = (170, 212, 0)  # teal BGR

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, label_text, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Stability filter
        current = set(detected)
        for obj in current:
            self.stability[obj] = self.stability.get(obj, 0) + 1
        for obj in list(self.stability.keys()):
            if obj not in current:
                self.stability[obj] = 0

        stable = [obj for obj, count in self.stability.items() if count >= self.STABLE_FRAMES]

        with self._lock:
            self.detections = stable.copy()
            self.detection_counts = dict(Counter(stable))

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==========================================
# TTS HELPER (Browser Web Speech API)
# ==========================================
def browser_speak(text, placeholder):
    if not text or text == st.session_state.last_spoken:
        return
    st.session_state.last_spoken = text
    escaped = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
    placeholder.empty()
    with placeholder:
        components.html(f"""
        <script>
            try {{
                window.speechSynthesis.cancel();
                var u = new SpeechSynthesisUtterance("{escaped}");
                u.rate = 0.85; u.pitch = 1.0; u.volume = 1.0;
                window.speechSynthesis.speak(u);
            }} catch(e) {{ console.log("TTS:", e); }}
        </script>
        """, height=0)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 👁️ Blind Assistant")
    st.caption("AI-powered vision & navigation")
    st.divider()

    # Detection settings
    st.markdown("### ⚙️ Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.10, 1.0, 0.40, 0.05)
    voice_enabled = st.toggle("🔊 Enable Voice Alerts", value=False,
                              help="Speaks detected objects using your browser's TTS")
    st.divider()

    # Navigation
    st.markdown("### 🗺️ Navigation")
    source = st.text_input("📍 Starting Point", placeholder="e.g., Coimbatore Railway Station")
    destination = st.text_input("🎯 Destination", placeholder="e.g., Gandhipuram Bus Stand")

    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        start_nav = st.button("🚶 Start", use_container_width=True)
    with nav_col2:
        stop_nav = st.button("⏹ Stop", use_container_width=True)

    if start_nav and source and destination:
        with st.spinner("🔍 Finding walking route..."):
            result, error = get_walking_directions(source, destination)
            if result:
                st.session_state.nav_steps = result["steps"]
                st.session_state.nav_summary = result["summary"]
                st.session_state.nav_current = 0
                st.session_state.nav_active = True
                st.session_state.last_spoken = ""
            else:
                st.error(f"❌ {error}")
    elif start_nav:
        st.warning("Please enter both locations")

    if stop_nav:
        st.session_state.nav_active = False
        st.session_state.nav_steps = []
        st.session_state.nav_current = 0

    # Show route summary
    if st.session_state.nav_active and st.session_state.nav_summary:
        summary = st.session_state.nav_summary
        st.success(f"🚶 {summary['distance']} • ⏱ {summary['duration']}")

        st.markdown("### 📋 Route Steps")
        for i, step in enumerate(st.session_state.nav_steps):
            if i == st.session_state.nav_current:
                st.markdown(f"**▶ Step {i+1}: {step['text']}**")
            elif i < st.session_state.nav_current:
                st.caption(f"✅ ~~{step['text']}~~")
            else:
                st.caption(f"⬜ {step['text']}")

        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("⬅ Prev", use_container_width=True):
                if st.session_state.nav_current > 0:
                    st.session_state.nav_current -= 1
                    st.session_state.last_spoken = ""
                    st.rerun()
        with next_col:
            if st.button("Next ➡", use_container_width=True):
                if st.session_state.nav_current < len(st.session_state.nav_steps) - 1:
                    st.session_state.nav_current += 1
                    st.session_state.last_spoken = ""
                    st.rerun()


# ==========================================
# MAIN CONTENT
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>👁️ Blind Assistant</h1>
    <p>Real-time AI object detection & walking navigation — accessible from any browser</p>
</div>
""", unsafe_allow_html=True)

col_cam, col_info = st.columns([3, 2])

# Camera Feed
with col_cam:
    st.markdown("### 📷 Live Camera")
    ctx = webrtc_streamer(
        key="blind-assistant",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=BlindAssistantProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Info Panel
with col_info:
    st.markdown("### 🎯 Detected Objects")
    detection_placeholder = st.empty()
    st.markdown("---")

    # Navigation current step
    if st.session_state.nav_active and st.session_state.nav_steps:
        idx = st.session_state.nav_current
        total = len(st.session_state.nav_steps)
        st.markdown("### 🧭 Navigation")
        st.progress((idx + 1) / total, text=f"Step {idx + 1} of {total}")
        step = st.session_state.nav_steps[idx]
        st.markdown(f"""<div class="nav-card nav-active">
            <strong>🚶 {step['text']}</strong>
        </div>""", unsafe_allow_html=True)

    tts_placeholder = st.empty()


# ==========================================
# REAL-TIME DETECTION DISPLAY LOOP
# ==========================================
if ctx.state.playing:
    if ctx.video_processor:
        with ctx.video_processor._lock:
            detections = ctx.video_processor.detection_counts.copy()

        ctx.video_processor.confidence = confidence

        with detection_placeholder.container():
            if detections:
                for obj, count in detections.items():
                    if obj in VEHICLE_OBJECTS:
                        css_class = "danger"
                        icon = "🚗"
                    elif obj in WARNING_OBJECTS:
                        css_class = "warn"
                        icon = "⚠️"
                    else:
                        css_class = ""
                        icon = "🟢"

                    label = f"{count} {obj}s" if count > 1 else obj
                    st.markdown(
                        f"""<div class="det-item {css_class}">
                        <span style="font-size:1.5rem">{icon}</span>
                        <span style="font-weight:500;font-size:1rem">{label} ahead</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                if voice_enabled:
                    parts = []
                    for obj, count in detections.items():
                        parts.append(
                            f"{count} {obj}s ahead" if count > 1 else f"{obj} ahead"
                        )
                    speak_text = ", ".join(parts)
                    browser_speak(speak_text, tts_placeholder)

            else:
                st.info("👀 No objects detected — point your camera at surroundings")

else:
    with detection_placeholder.container():
        st.markdown(
            """
        <div class="metric-box">
            <div class="value">📷</div>
            <div class="label">Click START above to activate the camera</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
