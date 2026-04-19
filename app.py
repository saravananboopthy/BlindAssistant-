import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_geolocation import streamlit_geolocation
import av
import cv2
import threading
import time
import math
import os
import googlemaps
import re
import json
from ultralytics import YOLO
from dotenv import load_dotenv

# Load ENV if available
load_dotenv()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

st.markdown("""
<style>
.main-header { background: linear-gradient(135deg, #00d4aa 0%, #667eea 100%); padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px; }
.det-card { padding: 12px; border-radius: 10px; margin-bottom: 8px; background: rgba(0,212,170,0.1); border-left: 5px solid #00d4aa; font-weight: bold; }
.danger { background: rgba(255,107,107,0.1); border-left-color: #ff6b6b; color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION & MODEL
# ==========================================
if "nav_steps" not in st.session_state:
    st.session_state.update({"nav_steps": [], "nav_idx": 0, "dest": "", "engine_active": False})

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==========================================
# NAVIGATION LOGIC
# ==========================================
def clean_html(text):
    return re.sub(r"<.*?>", "", text).replace("&nbsp;", " ").strip()

def get_walking_directions(source, dest, api_key):
    if not api_key: return None, "Missing API Key"
    try:
        gmaps = googlemaps.Client(key=api_key)
        res = gmaps.directions(source, dest, mode="walking")
        if not res: return None, "No walking route found."
        leg = res[0]["legs"][0]
        steps = []
        for s in leg["steps"]:
            instr = clean_html(s["html_instructions"])
            steps.append({
                "text": f"{instr} for {s['distance']['text']}",
                "lat": s["end_location"]["lat"],
                "lng": s["end_location"]["lng"]
            })
        return steps, None
    except Exception as e: return None, str(e)

# ==========================================
# VISION PROCESSOR
# ==========================================
class BlindProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if model:
            results = model(img, conf=0.45, verbose=False)[0]
            detected = []
            h, w, _ = img.shape
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                pos = "left" if (x1+x2)/2 < w*0.33 else "right" if (x1+x2)/2 > w*0.66 else "ahead"
                dist = "near" if (x2-x1) > 280 else "far"
                detected.append({"label": label, "pos": pos, "dist": dist})
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            with self.lock: self.detections = detected
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI & LOCATION
# ==========================================
st.markdown('<div class="main-header"><h1>👁️ Blind Assistant</h1></div>', unsafe_allow_html=True)

# Correct Path Handling for Streamlit Cloud
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(PARENT_DIR, "voice_engine")

try:
    my_voice_engine = components.declare_component("my_voice_engine", path=VOICE_DIR)
except Exception as e:
    st.error(f"Voice Engine Load Error: {e}")

location = streamlit_geolocation()
user_lat = location.get('latitude') if location else None
user_lng = location.get('longitude') if location else None

col_v, col_i = st.columns([1.5, 1])

current_detections = []
p_lat = float(user_lat) if user_lat else None
p_lng = float(user_lng) if user_lng else None

with col_v:
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=BlindProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    if ctx.video_processor:
        st.session_state.engine_active = st.toggle("Activate AI Voice Engine", value=st.session_state.engine_active)
        with ctx.video_processor.lock:
            current_detections = ctx.video_processor.detections.copy()

# RENDER MASTER ENGINE
my_voice_engine(
    detections=current_detections,
    nav_steps=st.session_state.nav_steps,
    engine_active=st.session_state.engine_active,
    p_lat=p_lat,
    p_lng=p_lng,
    key="fixed_voice_engine"
)            

with col_i:
    st.subheader("📍 Sensors & Path")
    if user_lat: st.success(f"GPS Locked: {user_lat:.4f}, {user_lng:.4f}")
    else: st.warning("Allow GPS (Click crosshair)")

    default_api = os.getenv("GOOGLE_MAPS_API_KEY", "")
    try:
        if st.secrets.get("GOOGLE_MAPS_API_KEY"): default_api = st.secrets["GOOGLE_MAPS_API_KEY"]
    except: pass
    
    api = st.text_input("G-Maps Key", value=default_api, type="password")
    dest_in = st.text_input("Destination", placeholder="e.g. NGP College or Hope College")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Start Navigation", type="primary", use_container_width=True):
            if user_lat and api and dest_in:
                source = f"{user_lat},{user_lng}"
                steps, error = get_walking_directions(source, dest_in, api)
                if steps:
                    st.session_state.update({"nav_steps": steps, "nav_idx": 0})
                    st.success("Route Found!")
                else: 
                    st.error(f"Navigation Failure: {error}")
                    st.info("Try a more specific destination (e.g., 'Hope College, Coimbatore')")
    
    with col_btn2:
        if st.button("Clear Route", use_container_width=True):
            st.session_state.nav_steps = []
            st.session_state.nav_idx = 0
            st.rerun()

    if st.session_state.nav_steps:
        idx = st.session_state.nav_idx
        st.info(f"**Step {idx+1}:** {st.session_state.nav_steps[idx]['text']}")
        if st.button("Next Step"):
            if st.session_state.nav_idx < len(st.session_state.nav_steps)-1:
                st.session_state.nav_idx += 1
                st.rerun()

    st.subheader("🎯 Live Detections")
    for d in current_detections:
        st.markdown(f'<div class="det-card {"danger" if d["dist"]=="near" else ""}">{d["label"].upper()} {d["pos"]}</div>', unsafe_allow_html=True)

if st.session_state.engine_active:
    time.sleep(1)
    st.rerun()
