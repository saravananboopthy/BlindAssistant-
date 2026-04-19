import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import threading
import time
import math
import os
import googlemaps
import re
import json
from collections import Counter
from ultralytics import YOLO
from streamlit_geolocation import streamlit_geolocation

# ==========================================
# PAGE SETTINGS & STYLES
# ==========================================
st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

st.markdown("""
<style>
.main-header { background: #1e293b; padding: 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px; }
.stButton>button { width: 100% !important; background-color: #ef4444; color: white; font-weight: bold; padding: 20px; font-size: 20px; border-radius: 15px; }
.stButton>button:hover { background-color: #dc2626; color: white; }
.active-btn { background-color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "nav_steps" not in st.session_state:
    st.session_state.update({"nav_steps": [], "nav_idx": 0, "nav_active": False, "last_nav": ""})

# ==========================================
# AI VISION ENGINE
# ==========================================
@st.cache_resource
def load_yolo(): return YOLO("yolov8n.pt")
model = load_yolo()

class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model:
            res = model(img, conf=0.45, verbose=False)[0]
            now = []
            h, w, _ = img.shape
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                label = model.names[int(b.cls[0])]
                # Position logic
                pos = "left" if (x1+x2)/2 < w*0.33 else "right" if (x1+x2)/2 > w*0.66 else "ahead"
                now.append({"label": label, "pos": pos})
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            with self.lock: self.detections = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# NAVIGATION CORE
# ==========================================
def get_walking_directions(source, dest, api_key):
    try:
        gmaps = googlemaps.Client(key=api_key)
        res = gmaps.directions(source, dest, mode="walking")
        if not res: return None, "No path found."
        leg = res[0]["legs"][0]
        steps = []
        for s in leg["steps"]:
            instr = re.sub(r"<.*?>", "", s["html_instructions"]).replace("&nbsp;", " ").strip()
            steps.append({"text": instr, "lat": s["end_location"]["lat"], "lng": s["end_location"]["lng"]})
        return steps, None
    except Exception as e: return None, str(e)

# ==========================================
# UI & INPUTS
# ==========================================
st.markdown('<div class="main-header"><h1>👁️ Blind Assistant</h1></div>', unsafe_allow_html=True)

# Hidden Geolocation (for background tracking)
location = streamlit_geolocation()
u_lat = location.get('latitude') if location else None
u_lng = location.get('longitude') if location else None

col_cam, col_ctrl = st.columns([1.5, 1])

# Data to pass to the JS Control Hub
nav_data = {"current": "", "goal": None}
if st.session_state.nav_active and st.session_state.nav_idx < len(st.session_state.nav_steps):
    step = st.session_state.nav_steps[st.session_state.nav_idx]
    nav_data["current"] = step["text"]
    nav_data["goal"] = step

object_data = []

with col_cam:
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )
    if ctx.video_processor:
        with ctx.video_processor.lock:
            objs = ctx.video_processor.detections.copy()
        if objs:
            for o in objs:
                object_data.append(f"{o['label']} on your {o['pos']}")

with col_ctrl:
    st.subheader("Settings")
    
    # Auto-load API from Secrets
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", os.getenv("GOOGLE_MAPS_API_KEY", ""))
    dest_in = st.text_input("Destination", placeholder="e.g. Hospital or Hope College", key="dest_field")

    # The Master Unlock Button (Essential for voice on web)
    # We combine unlocking voice with starting navigation to ensure a user gesture
    if st.button("🚀 ACTIVATE ASSISTANT & VOICE"):
        if u_lat and api_key and dest_in:
            steps, err = get_walking_directions((u_lat, u_lng), dest_in, api_key)
            if steps:
                st.session_state.update({"nav_steps": steps, "nav_idx": 0, "nav_active": True})
            else: st.error(err)
        else:
            st.warning("Please allow GPS location (click the crosshair icon) and enter a destination.")

    if st.session_state.nav_active:
        st.success(f"**Current Path:** {nav_data['current']}")
        if st.button("Stop"): st.session_state.nav_active = False; st.rerun()

# ==========================================
# MASTER JS BRAIN (Handles Speech, Memory, & Auto-Navigation)
# ==========================================
# This script survived reloads by using internal state
master_js = f"""
<div id="status-box" style="background:#f1f5f9; padding:15px; border-radius:10px; text-align:center; font-family:sans-serif; border:1px solid #cbd5e1;">
    <b style="color:#1e293b;">System State:</b> <span id="sys-msg" style="color:#ef4444;">Voice Locked</span><br>
    <small style="color:#64748b;">(Voice starts after you click the Activate button)</small>
</div>
<script>
    // 1. Initial State
    window.voiceEnabled = window.voiceEnabled || false;
    const parent = window.parent;
    let data = {json.dumps({"nav": nav_data["current"], "nav_goal": nav_data["goal"], "alert": object_data})};

    function speak(t, priority=false) {{
        if (!window.voiceEnabled) return;
        if (window.speechSynthesis.speaking && !priority) return;
        if (priority) window.speechSynthesis.cancel();
        let u = new SpeechSynthesisUtterance(t);
        u.rate = 1.0;
        window.speechSynthesis.speak(u);
    }}

    // 2. Unlock Logic (Triggers when the iframe is first interacted with or re-rendered after button)
    if (!window.voiceEnabled) {{
        // We attempt to unlock on any mouse movement in the box as well
        document.body.onmouseover = () => {{
            if (!window.voiceEnabled) {{
                window.voiceEnabled = true;
                document.getElementById('sys-msg').innerText = 'Voice Ready';
                document.getElementById('sys-msg').style.color = '#10b981';
                speak("Voice Assistant Online");
            }}
        }};
    }}

    // 3. Navigation Voice (Only speak if changed)
    if (data.nav && sessionStorage.getItem('last_spoken_nav') !== data.nav) {{
        speak("Navigation Update: " + data.nav, true);
        sessionStorage.setItem('last_spoken_nav', data.nav);
    }}

    // 4. Object Detection Voice (With Spatial Awareness & Throttling)
    if (data.alert.length > 0 && window.voiceEnabled) {{
        let msg = "Ahead is " + data.alert.join(" and ");
        let now = Date.now();
        let lastObjMsg = sessionStorage.getItem('last_obj_msg');
        let lastObjTime = parseInt(sessionStorage.getItem('last_obj_time') || "0");

        // 8 second silence between same objects
        if (lastObjMsg !== msg || (now - lastObjTime > 8000)) {{
            speak(msg, false);
            sessionStorage.setItem('last_obj_msg', msg);
            sessionStorage.setItem('last_obj_time', now.toString());
        }}
    }}
</script>
"""
components.html(master_js, height=120)

# Refresh Loop
time.sleep(1.5)
st.rerun()
