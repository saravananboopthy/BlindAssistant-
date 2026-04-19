import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_geolocation import streamlit_geolocation
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import threading
import time
import math
import googlemaps
import re
import os
import json
from collections import Counter
from ultralytics import YOLO

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

# Stabilized Page Refresh
st_autorefresh(interval=1500, key="global_sync_refresh")

# Session State
if "nav_steps" not in st.session_state:
    st.session_state.update({
        "nav_steps": [], "nav_idx": 0, "nav_active": False, 
        "last_nav": "", "detected_memory": {}
    })

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
            steps.append({
                "text": instr,
                "lat": s["end_location"]["lat"], "lng": s["end_location"]["lng"]
            })
        return steps, None
    except Exception as e: return None, str(e)

# ==========================================
# AI VISION
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
        res = model(img, conf=0.45, verbose=False)[0]
        now = []
        h, w, _ = img.shape
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            label = model.names[int(b.cls[0])]
            pos = "on your left" if (x1+x2)/2 < w*0.33 else "on your right" if (x1+x2)/2 > w*0.66 else "straight ahead"
            now.append({"label": label, "pos": pos})
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        with self.lock: self.detections = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI DESIGN
# ==========================================
st.title("👁️ Blind Assistant")

location = streamlit_geolocation()
u_lat = location.get('latitude') if location else None
u_lng = location.get('longitude') if location else None

col1, col2 = st.columns([2, 1])

# DATA FOR THE MASTER VOICE & NAV ENGINE
current_nav_step = ""
current_nav_data = []
if st.session_state.nav_active:
    idx = st.session_state.nav_idx
    if idx < len(st.session_state.nav_steps):
        current_nav_step = st.session_state.nav_steps[idx]["text"]
        current_nav_data = st.session_state.nav_steps[idx]

detection_alerts = []

with col1:
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("🧭 Path & Settings")
    
    # Check for Step Advance Request from JS
    if st.query_params.get("step_reached"):
        if st.session_state.nav_idx < len(st.session_state.nav_steps) - 1:
            st.session_state.nav_idx += 1
            st.query_params.clear()
            st.rerun()

    default_api = os.getenv("GOOGLE_MAPS_API_KEY", "")
    try:
        if st.secrets.get("GOOGLE_MAPS_API_KEY"): default_api = st.secrets["GOOGLE_MAPS_API_KEY"]
    except: pass
    
    api_key = st.text_input("G-Maps Key", value=default_api, type="password")
    dest = st.text_input("Destination", placeholder="e.g. Hope College")

    if st.button("Start Assistant", type="primary", use_container_width=True):
        if u_lat and api_key and dest:
            data, err = get_walking_directions((u_lat, u_lng), dest, api_key)
            if data:
                st.session_state.update({"nav_steps": data, "nav_idx": 0, "nav_active": True})
            else: st.error(err)

    if st.session_state.nav_active:
        st.info(f"**Step {st.session_state.nav_idx + 1}:** {current_nav_step}")
        if st.button("Stop"): st.session_state.nav_active = False; st.rerun()

    st.subheader("🎯 Active Detections")
    if ctx.video_processor:
        with ctx.video_processor.lock:
            objs = ctx.video_processor.detections.copy()
        if objs:
            st.write(", ".join([f"{o['label']} ({o['pos']})" for o in objs]))
            now = time.time()
            for o in objs:
                tag = f"{o['label']}_{o['pos']}"
                if tag not in st.session_state.detected_memory or now - st.session_state.detected_memory[tag] > 8:
                    detection_alerts.append(f"{o['label']} {o['pos']}")
                    st.session_state.detected_memory[tag] = now

# ==========================================
# MASTER SMART ENGINE (Auto-Nav + Positional Voice)
# ==========================================
js_engine = f"""
    <div style="background:#f8fafc; padding:15px; border-radius:10px; text-align:center; border:2px solid #e2e8f0; font-family:sans-serif;">
        <button id="vbtn" onclick="ulock()" style="background:#ef4444; color:white; border:none; padding:10px; border-radius:8px; cursor:pointer; font-weight:bold; width:100%;">
            🚨 TAP TO ENABLE VOICE & AUTO-GPS 🔊
        </button>
        <div id="status" style="margin-top:8px; font-size:12px; color:#64748b;">System Standby</div>
    </div>
    <script>
    const data = {json.dumps({"nav": current_nav_step, "nav_goal": current_nav_data, "objs": detection_alerts})};
    
    function speak(t, priority=false) {{
        if (!window.active) return;
        if (priority) window.speechSynthesis.cancel();
        let u = new SpeechSynthesisUtterance(t); u.rate = 1.0;
        window.speechSynthesis.speak(u);
    }}

    function ulock() {{
        window.active = true;
        let b = document.getElementById('vbtn');
        b.style.background = '#10b981'; b.innerText = '✔️ ASSISTANT LIVE';
        document.getElementById('status').innerText = 'GPS Tracking & AI Detections Active';
        speak("System ready. Vision and Navigation online.");
        
        // Start HIGH-ACCURACY WatchPosition inside the engine
        navigator.geolocation.watchPosition((p) => {{
            window.curLat = p.coords.latitude; window.curLng = p.coords.longitude;
            checkProximity();
        }}, (e) => {{ console.error(e); }}, {{enableHighAccuracy: true}});
    }}

    function checkProximity() {{
        if (data.nav_goal && data.nav_goal.lat) {{
            const R = 6371e3;
            const dLat = (data.nav_goal.lat - window.curLat) * Math.PI/180;
            const dLng = (data.nav_goal.lng - window.curLng) * Math.PI/180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(window.curLat * Math.PI/180) * Math.cos(data.nav_goal.lat * Math.PI/180) * Math.sin(dLng/2) * Math.sin(dLng/2);
            const dist = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            
            // If within 15 meters, tell Python to advance to next step
            if (dist < 15) {{
                window.parent.postMessage({{type: 'streamlit:set_query_params', queryParams: {{'step_reached': Date.now()}} }}, '*');
            }}
        }}
    }}

    // Trigger Navigation Voice
    if (data.nav && sessionStorage.getItem('last_nav') !== data.nav) {{
        speak("Next: " + data.nav, true);
        sessionStorage.setItem('last_nav', data.nav);
    }}

    // Trigger Object Voice with Placement
    if (data.objs.length > 0) {{
        let msg = "I see " + data.objs.join(" and ");
        speak(msg, false);
    }}
    </script>
"""
components.html(js_engine, height=130)
