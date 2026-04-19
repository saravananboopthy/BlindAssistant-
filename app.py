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
    return YOLO("yolov8n.pt")

model = load_yolo()

class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model:
            res = model(img, conf=0.4, verbose=False)[0]
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

# UI Logic
user_lat = st.query_params.get("lat", None)
user_lng = st.query_params.get("lng", None)

st.title("👁️ Blind Assistant")
st.write(f"GPS Status: {'🟢 Active' if user_lat else '🔴 Searching...'}")

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

# THE PERSISTENT CONTROLLER
js_code = f"""
    <div style="background:#f0f2f6; padding:10px; border-radius:10px; font-family:sans-serif; display:flex; gap:10px; align-items:center;">
        <button id="v-btn" style="background:#667eea; color:white; border:none; padding:8px 12px; border-radius:5px; cursor:pointer;">🎤 Search</button>
        <span style="font-size:0.8rem;">Voice engine: {'ON' if st.session_state.engine_active else 'OFF'}</span>
    </div>
    <script>
    const parent = window.parent;
    if (!window.init) {{
        navigator.geolocation.watchPosition((p) => {{
            parent.postMessage({{ type: 'streamlit:set_query_params', queryParams: {{ lat: p.coords.latitude.toFixed(6), lng: p.coords.longitude.toFixed(6) }} }}, '*');
        }}, (e) => console.log(e), {{ enableHighAccuracy: true }});
        window.init = true;
    }}
    const recBtn = document.getElementById('v-btn');
    if ('webkitSpeechRecognition' in window) {{
        const r = new webkitSpeechRecognition();
        recBtn.onclick = () => r.start();
        r.onresult = (e) => parent.postMessage({{ type: 'streamlit:set_query_params', queryParams: {{ dest: e.results[0][0].transcript }} }}, '*');
    }}
    function speak(t, prio=false) {{
        if (window.speechSynthesis.speaking && !prio) return;
        if (prio) window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(t); u.rate = 0.9;
        window.speechSynthesis.speak(u);
    }}
    const dets = {json.dumps(current_dets)};
    const nav = "{nav_msg}";
    if ({str(st.session_state.engine_active).lower()}) {{
        if (nav && window.lNav !== nav) {{ speak("Instruction: " + nav, true); window.lNav = nav; }}
        else if (dets.length > 0) {{
            const d = dets[0]; const m = d.label + " " + d.pos;
            if (window.lDet !== m) {{ speak(m, d.dist === 'near'); window.lDet = m; }}
        }}
    }}
    </script>
"""
components.html(js_code, height=60)

with col_i:
    st.subheader("Navigation")
    api = st.text_input("G-Maps Key", type="password")
    dest = st.text_input("Destination", value=st.query_params.get("dest", ""))
    if st.button("Start"):
        if user_lat and api and dest:
            try:
                g = googlemaps.Client(key=api)
                r = g.directions((float(user_lat), float(user_lng)), dest, mode="walking")
                if r:
                    steps = [s["html_instructions"].replace("<b>","").replace("</b>","") for s in r[0]["legs"][0]["steps"]]
                    st.session_state.update({"nav_steps": steps, "nav_idx": 0})
                else: st.error("No path")
            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.nav_steps:
        st.info(st.session_state.nav_steps[st.session_state.nav_idx])
        if st.button("Next Step"):
            if st.session_state.nav_idx < len(st.session_state.nav_steps)-1:
                st.session_state.nav_idx += 1
                st.rerun()

    if current_dets:
        for d in current_dets:
            st.markdown(f'<div class="det-card"><b>{d["label"].upper()}</b> {d["pos"]}</div>', unsafe_allow_html=True)

if st.session_state.engine_active:
    time.sleep(1)
    st.rerun()
