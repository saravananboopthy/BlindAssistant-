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
from streamlit_geolocation import streamlit_geolocation

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

# Custom CSS
st.markdown("""
<style>
.det-card { padding: 10px; border-radius: 8px; margin-bottom: 5px; background: rgba(0,212,170,0.1); border-left: 5px solid #00d4aa; }
.danger { background: rgba(255,107,107,0.1); border-left-color: #ff6b6b; color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "nav_steps" not in st.session_state:
    st.session_state.update({"nav_steps": [], "nav_idx": 0, "dest": "", "engine_active": False})

@st.cache_resource
def load_yolo():
    # Upgrade to 's' model for better accuracy
    return YOLO("yolov8s.pt")

model = load_yolo()

class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model:
            res = model(img, conf=0.5, verbose=False)[0] 
            now = []
            h, w, _ = img.shape
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                label = model.names[int(b.cls[0])]
                pos = "left" if (x1+x2)/2 < w*0.33 else "right" if (x1+x2)/2 > w*0.66 else "ahead"
                dist = "near" if (x2-x1) > 280 else "far"
                now.append({"label": label, "pos": pos, "dist": dist})
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,212,170), 2)
            with self.lock: self.detections = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("👁️ Blind Assistant")

# Real Geolocation fetching
st.write("### 📍 Location")
loc_col, stat_col = st.columns([1, 2])
with loc_col:
    location = streamlit_geolocation()

user_lat = location.get('latitude') if location else None
user_lng = location.get('longitude') if location else None

with stat_col:
    if user_lat:
        st.success(f"**GPS Locked:** {user_lat:.4f}, {user_lng:.4f}")
    else:
        st.error("**GPS Status:** Click the crosshair icon to lock your Starting Location")

# ==============================================================================
# JAVASCRIPT MASTER ENGINE (Official Streamlit Component Wrapper)
# ==============================================================================
my_voice_engine = components.declare_component("my_voice_engine", path="voice_engine")
# ==============================================================================

col_v, col_i = st.columns([1.5, 1])
current_dets = []

with col_v:
    ctx = webrtc_streamer(key="v", video_processor_factory=VisionProcessor, 
                         rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
    
    if ctx.video_processor:
        st.session_state.engine_active = st.toggle("Activate Voice engine", value=st.session_state.engine_active)
        with ctx.video_processor.lock: 
            current_dets = ctx.video_processor.detections.copy()

# Render the TRUE custom component (this iframe NEVER dies during reruns)
my_voice_engine(
    detections=current_dets, 
    nav_steps=st.session_state.nav_steps, 
    engine_active=st.session_state.engine_active,
    key="engine_component"
)            


with col_i:
    st.subheader("🧭 Path Input")
    
    # Safely load the API Key
    default_api = os.getenv("GOOGLE_MAPS_API_KEY", "")
    try:
        if st.secrets.get("GOOGLE_MAPS_API_KEY"):
            default_api = st.secrets["GOOGLE_MAPS_API_KEY"]
    except: pass
    
    api = st.text_input("G-Maps Key", value=default_api, type="password")
    
    # User uses built-in OS Voice dictation on this text field
    st.info("💡 Tap the text box below and use your phone's built-in 🎤 microphone button to speak your destination.")
    dest = st.text_input("Destination", placeholder="e.g. Hospital")
    
    if st.button("Start Navigation", type="primary"):
        if user_lat and api and dest:
            try:
                g = googlemaps.Client(key=api)
                r = g.directions((float(user_lat), float(user_lng)), dest, mode="walking")
                if r:
                    steps = []
                    for s in r[0]["legs"][0]["steps"]:
                        text = s["html_instructions"].replace("<b>","").replace("</b>","")
                        lat = s["end_location"]["lat"]
                        lng = s["end_location"]["lng"]
                        steps.append({"text": text, "lat": lat, "lng": lng})
                    
                    st.success("Route Generated! Voice Navigation Active.")
                    # Inject steps into localstorage for engine
                    inj_steps = f"""<script>
                    localStorage.setItem('nav_steps', '{json.dumps(steps)}');
                    localStorage.setItem('nav_idx', '0');
                    </script>"""
                    components.html(inj_steps, height=0)
                else: 
                    st.error("No path found.")
            except Exception as e: 
                st.error(f"Error: {e}")

    st.subheader("🎯 Active Detections")
    if 'current_dets' in locals() and current_dets:
        for d in current_dets:
            st.markdown(f'<div class="det-card {"danger" if d["dist"]=="near" else ""} "><b>{d["label"].upper()}</b> {d["pos"]}</div>', unsafe_allow_html=True)
            
if st.session_state.engine_active:
    time.sleep(1)
    st.rerun()
