import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import threading
import time
import os
import googlemaps
from ultralytics import YOLO
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

# ==========================================
# PAGE CONFIG & CSS
# ==========================================
st.set_page_config(
    page_title="Blind Assistant | Cloud Edition",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(102,126,234,0.08));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.main-header h1 {
    background: linear-gradient(135deg, #00d4aa, #667eea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem;
}
.nav-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(0,212,170,0.05));
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0;
}
.det-item {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.6rem 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;
    background: rgba(0,212,170,0.06); border-left: 3px solid #00d4aa;
}
.det-item.warn { background: rgba(255,107,107,0.06); border-left-color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# UTILS & MODEL
# ==========================================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ==========================================
# VIDEO PROCESSOR (AI ENGINE)
# ==========================================
class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
        self.debounce = {}
        self.last_spoken = {}
        self.last_global_speak = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.35, verbose=False)[0]
        
        h, w, _ = img.shape
        detected_now = []
        current_labels = set()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            
            x_center = (x1 + x2) / 2
            if x_center < w * 0.33: pos = "on your Left"
            elif x_center > w * 0.66: pos = "on your Right"
            else: pos = "Immediately Ahead"

            dist = "near" if (x2 - x1) > 250 else "far"
            detected_now.append({"label": label, "pos": pos, "dist": dist})
            current_labels.add(label)

            # Draw
            color = (107, 107, 255) if dist == "near" else (170, 212, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {dist}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with self.lock:
            self.detections = detected_now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# SESSION STATE
# ==========================================
if "nav_steps" not in st.session_state:
    st.session_state.update({
        "nav_steps": [], "nav_idx": 0, "destination": "", 
        "last_spoken": "", "speak_queue": []
    })

# ==========================================
# JS TTS HELPER
# ==========================================
def browser_speak(text, rate=0.8):
    if not text: return
    # Sanitize text
    clean_text = text.replace("'", "").replace('"', "")
    components.html(f"""
        <script>
            var u = new SpeechSynthesisUtterance('{clean_text}');
            u.rate = {rate};
            window.speechSynthesis.speak(u);
        </script>
    """, height=0)

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>👁️ Blind Assistant - Cloud Edition</h1>
    <p>Real-time AI Vision & Walking Navigation (Web Standalone)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Controls")
    api_key = st.text_input("Google Maps API Key", value=os.getenv("GOOGLE_MAPS_API_KEY", ""), type="password")
    
    st.divider()
    dest_input = st.text_input("📍 Set Destination", placeholder="e.g. Times Square")
    if st.button("Start Navigation") and dest_input and api_key:
        try:
            gmaps = googlemaps.Client(key=api_key)
            # Simple Geocode for now
            res = gmaps.directions("current location", dest_input, mode="walking")
            if res:
                leg = res[0]["legs"][0]
                steps = []
                for s in leg["steps"]:
                    steps.append(s["html_instructions"].replace("<b>", "").replace("</b>", ""))
                st.session_state.nav_steps = steps
                st.session_state.nav_idx = 0
                st.session_state.destination = dest_input
                st.success("Route Found!")
                browser_speak(f"Navigating to {dest_input}")
        except Exception as e:
            st.error(f"Error: {e}")

col_vid, col_info = st.columns([3, 2])

with col_vid:
    webrtc_ctx = webrtc_streamer(
        key="vision",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )

with col_info:
    st.subheader("🧭 Navigation Step")
    if st.session_state.nav_steps:
        idx = st.session_state.nav_idx
        step_text = st.session_state.nav_steps[idx]
        st.markdown(f"""<div class="nav-card"><h3>Step {idx+1}</h3><p>{step_text}</p></div>""", unsafe_allow_html=True)
        if st.button("Next Step"):
            st.session_state.nav_idx += 1
            browser_speak(st.session_state.nav_steps[st.session_state.nav_idx])
    else:
        st.info("Set destination in sidebar to begin.")

    st.divider()
    st.subheader("🎯 Active Detections")
    det_placeholder = st.empty()

# ==========================================
# MAIN LOOP (TTS & UPDATE)
# ==========================================
if webrtc_ctx.video_processor:
    while True:
        with webrtc_ctx.video_processor.lock:
            current_detections = webrtc_ctx.video_processor.detections.copy()
        
        with det_placeholder.container():
            for d in current_detections:
                is_near = d['dist'] == "near"
                css = "warn" if is_near else ""
                st.markdown(f"""<div class="det-item {css}"><b>{d['label'].upper()}</b> {d['pos']} ({d['dist']})</div>""", unsafe_allow_html=True)
        
        # Simple Logic to avoid repeat speech
        if current_detections:
            top_obj = current_detections[0]
            speak_msg = f"{top_obj['label']} {top_obj['pos']}"
            if speak_msg != st.session_state.last_spoken:
                # Add a 3 second cooldown for repetitive object announcements in UI
                st.session_state.last_spoken = speak_msg
                browser_speak(speak_msg, rate=0.7) # Using SLOW rate as requested!
        
        time.sleep(1.5) # Global gap between detection checks