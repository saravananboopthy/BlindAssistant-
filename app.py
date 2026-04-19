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
            res = model(img, conf=0.5, verbose=False)[0] # Increased confidence to prevent wrong detections
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
st.write("### 📍 Location & Navigation")
loc_col, stat_col = st.columns([1, 2])
with loc_col:
    location = streamlit_geolocation()

user_lat = location.get('latitude') if location else None
user_lng = location.get('longitude') if location else None

with stat_col:
    st.write(f"**GPS Status:** {'🟢 Active' if user_lat else '🔴 Click the crosshair icon to get Location'}")
    if user_lat:
        st.write(f"Lat: {user_lat:.4f}, Lng: {user_lng:.4f}")

col_v, col_i = st.columns([1.5, 1])
with col_v:
    ctx = webrtc_streamer(key="v", video_processor_factory=VisionProcessor, 
                         rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
    if ctx.video_processor:
        st.session_state.engine_active = st.toggle("Activate Voice engine", value=st.session_state.engine_active)

# Data for JS
current_dets = []
if ctx.video_processor:
    with ctx.video_processor.lock: current_dets = ctx.video_processor.detections.copy()

nav_msg = st.session_state.nav_steps[st.session_state.nav_idx] if st.session_state.nav_steps else ""

# THE PERSISTENT VOICE CONTROLLER (Throttled so it doesn't repeat constantly)
js_code = f"""
    <div style="background:#f0f2f6; padding:10px; border-radius:10px; font-family:sans-serif; text-align:center;">
        <span style="font-size:0.8rem;"><b>Voice Engine Status:</b> {'ON 🔊' if st.session_state.engine_active else 'OFF 🔇'}</span>
    </div>
    <script>
    function speak(t, prio=false) {{
        if (window.speechSynthesis.speaking && !prio) return;
        if (prio) window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(t); u.rate = 0.9;
        window.speechSynthesis.speak(u);
    }}
    
    if (!window.voiceInit) {{
        window.lDet = "";
        window.lDetTime = 0;
        window.voiceInit = true;
    }}

    const dets = {json.dumps(current_dets)};
    const nav = "{nav_msg}";
    const now = Date.now();

    if ({str(st.session_state.engine_active).lower()}) {{
        if (nav && window.lNav !== nav) {{ 
            speak("Instruction: " + nav, true); 
            window.lNav = nav; 
        }}
        else if (dets.length > 0) {{
            const d = dets[0]; const m = d.label + " " + d.pos;
            // Only speak if it's a new message OR 8 seconds have passed since we last said it
            if (window.lDet !== m || (now - window.lDetTime) > 8000) {{ 
                speak(m, d.dist === 'near'); 
                window.lDet = m; 
                window.lDetTime = now;
            }}
        }}
    }}
    </script>
"""
components.html(js_code, height=60)

with col_i:
    st.subheader("🧭 Path Input")
    api = st.text_input("G-Maps Key", type="password")
    
    # Destination Input: We recommend using the device's default microphone/dictation here.
    st.info("💡 Tap the text box below and use your phone/computer's built-in 🎤 microphone button to speak your destination.")
    dest = st.text_input("Destination", placeholder="e.g. Hospital")
    
    if st.button("Start Navigation", type="primary"):
        if user_lat and api and dest:
            try:
                g = googlemaps.Client(key=api)
                r = g.directions((float(user_lat), float(user_lng)), dest, mode="walking")
                if r:
                    steps = [s["html_instructions"].replace("<b>","").replace("</b>","") for s in r[0]["legs"][0]["steps"]]
                    st.session_state.update({"nav_steps": steps, "nav_idx": 0})
                else: st.error("No path found.")
            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.nav_steps:
        st.success("Routing active.")
        st.write(f"**Step {st.session_state.nav_idx + 1}:**")
        st.info(st.session_state.nav_steps[st.session_state.nav_idx])
        if st.button("Next Step ➡"):
            if st.session_state.nav_idx < len(st.session_state.nav_steps)-1:
                st.session_state.nav_idx += 1
                st.rerun()
            else:
                st.success("Destination Reached!")
                st.session_state.nav_steps = []

    st.subheader("🎯 Active Detections")
    if current_dets:
        for d in current_dets:
            st.markdown(f'<div class="det-card {"danger" if d["dist"]=="near" else ""} "><b>{d["label"].upper()}</b> {d["pos"]}</div>', unsafe_allow_html=True)

if st.session_state.engine_active:
    time.sleep(1)
    st.rerun()
