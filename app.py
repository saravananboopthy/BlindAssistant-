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
import json

# Load environment variables
load_dotenv()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Blind Assistant | Cloud Edition", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.det-item { 
    padding: 12px; border-radius: 10px; margin-bottom: 8px; 
    background: rgba(0,212,170,0.08); border-left: 5px solid #00d4aa; 
    font-weight: 500;
}
.warn { background: rgba(255,107,107,0.08); border-left-color: #ff6b6b; color: #ff6b6b; }
.main-header {
    background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(102,126,234,0.1));
    padding: 1rem 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "nav_steps" not in st.session_state:
    st.session_state.update({
        "nav_steps": [], "nav_idx": 0, "destination": "", 
        "engine_active": False
    })

# ==========================================
# AI MODEL LOADING
# ==========================================
@st.cache_resource
def load_yolo():
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

model = load_yolo()

# ==========================================
# VISION ENGINE
# ==========================================
class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if model:
            results = model(img, conf=0.4, verbose=False)[0]
            detected_now = []
            h, w, _ = img.shape
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                x_center = (x1 + x2) / 2
                pos = "on your left" if x_center < w * 0.33 else "on your right" if x_center > w * 0.66 else "ahead"
                dist = "near" if (x2 - x1) > 280 else "far"
                detected_now.append({"label": label, "pos": pos, "dist": dist})
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 212, 170), 2)
            with self.lock:
                self.detections = detected_now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI TOP BAR
# ==========================================
user_lat = st.query_params.get("lat")
user_lng = st.query_params.get("lng")
if st.query_params.get("dest"):
    st.session_state.destination = st.query_params.get("dest")

st.markdown(f"""
<div class="main-header">
    <h2 style="margin:0;">👁️ Blind Assistant</h2>
    <p style="margin:0; color:#666;">GPS: {'Active' if user_lat else 'Searching...'}</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# THE CONTROLLER (GPS + Unified TTS Queue)
# ==========================================
current_detections = []
nav_instruction = ""

# Getting latest data for the JS component
if "vision_ctx" in st.session_state and st.session_state.vision_ctx.video_processor:
    with st.session_state.vision_ctx.video_processor.lock:
        current_detections = st.session_state.vision_ctx.video_processor.detections.copy()

if st.session_state.nav_steps:
    nav_instruction = st.session_state.nav_steps[st.session_state.nav_idx]

# Inject persistent JS
components.html(f"""
    <div style="display: flex; gap: 10px; align-items: center; padding: 8px; background: #eef2f7; border-radius: 15px; font-family: sans-serif;">
        <button id="mic-btn" style="background: #667eea; color: white; border: none; padding: 8px 15px; border-radius: 10px; cursor: pointer; font-weight: 600;">
            🎤 Search Destination
        </button>
        <span id="gps-info" style="font-size: 0.8rem; color: #555;">GPS: {user_lat or 'Finding...'}</span>
    </div>

    <script>
    const parent = window.parent;
    
    // GPS TRACKER
    if (!window.gpsTracker) {{
        navigator.geolocation.watchPosition(
            (p) => {{
                parent.postMessage({{ 
                    type: 'streamlit:set_query_params', 
                    queryParams: {{ lat: p.coords.latitude.toFixed(6), lng: p.coords.longitude.toFixed(6) }} 
                }}, '*');
            }},
            (e) => console.error(e),
            {{ enableHighAccuracy: true }}
        );
        window.gpsTracker = true;
    }}

    // VOICE COMMAND
    if ('webkitSpeechRecognition' in window) {{
        const rec = new webkitSpeechRecognition();
        document.getElementById('mic-btn').onclick = () => rec.start();
        rec.onresult = (e) => {{
            parent.postMessage({{ type: 'streamlit:set_query_params', queryParams: {{ dest: e.results[0][0].transcript }} }}, '*');
        }};
    }}

    // UNIFIED VOICE ENGINE (Queued)
    function speak(text, priority=false) {{
        if (window.speechSynthesis.speaking && !priority) return;
        if (priority) window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 0.9;
        window.speechSynthesis.speak(u);
    }}

    const dets = {json.dumps(current_detections)};
    const nav = "{nav_instruction}";
    const active = {str(st.session_state.engine_active).lower()};

    if (active) {{
        // Priority 1: Navigation
        if (nav && window.lastNavStep !== nav) {{
            speak("Navigation updated: " + nav, true);
            window.lastNavStep = nav;
        }}
        // Priority 2: Safety Alerts
        else if (dets.length > 0) {{
            const d = dets[0];
            const msg = (d.dist === 'near' ? "Danger " : "") + d.label + " " + d.pos;
            if (window.lastDetMsg !== msg) {{
                speak(msg, d.dist === 'near');
                window.lastDetMsg = msg;
            }}
        }}
    }}
    </script>
""", height=60, key=f"voice_geo_engine_{time.time()}")

# ==========================================
# MAIN UI
# ==========================================
col_v, col_i = st.columns([1.8, 1])

with col_v:
    st.session_state.vision_ctx = webrtc_streamer(
        key="vision",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    if st.session_state.vision_ctx.video_processor:
        st.session_state.engine_active = st.toggle("🔊 Activate AI Voice Assistant", value=st.session_state.engine_active)

with col_i:
    st.subheader("🧭 Guidance")
    api_key = st.text_input("G-Maps Key", value=os.getenv("GOOGLE_MAPS_API_KEY", ""), type="password")
    dest_in = st.text_input("Target", value=st.session_state.destination)
    
    if st.button("🚀 Start", use_container_width=True):
        if user_lat and api_key and dest_in:
            try:
                gmaps = googlemaps.Client(key=api_key)
                res = gmaps.directions((float(user_lat), float(user_lng)), dest_in, mode="walking")
                if res:
                    leg = res[0]["legs"][0]
                    steps = [s["html_instructions"].replace("<b>", "").replace("</b>", "").replace('<div style="font-size:0.9em">', " ").replace("</div>", "") for s in leg["steps"]]
                    st.session_state.update({"nav_steps": steps, "nav_idx": 0, "destination": dest_in})
                else: st.error("No path found")
            except Exception as e: st.error(f"API Error: {e}")

    if st.session_state.nav_steps:
        curr = st.session_state.nav_idx
        st.markdown(f"**Step {curr+1}:** {st.session_state.nav_steps[curr]}")
        if st.button("Next Step ➡", use_container_width=True):
            if st.session_state.nav_idx < len(st.session_state.nav_steps) - 1:
                st.session_state.nav_idx += 1
                st.rerun()

    st.divider()
    st.subheader("🎯 Detection")
    if current_detections:
        for d in current_detections:
            st.markdown(f'<div class="det-item {"warn" if d["dist"]=="near" else ""}"><b>{d["label"].upper()}</b> {d["pos"]}</div>', unsafe_allow_html=True)
    else: st.write("Searching path...")

# Loop
if st.session_state.engine_active:
    time.sleep(1.0)
    st.rerun()
