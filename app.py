import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import threading
import time
import math
import googlemaps
import re
import json
import os
from collections import Counter
from ultralytics import YOLO
from streamlit_geolocation import streamlit_geolocation

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

# Session State for App Memory
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
        res = model(img, conf=0.45, verbose=False)[0]
        now = []
        h, w, _ = img.shape
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            label = model.names[int(b.cls[0])]
            pos = "left" if (x1+x2)/2 < w*0.33 else "right" if (x1+x2)/2 > w*0.66 else "ahead"
            now.append(f"{label} {pos}")
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
# UI
# ==========================================
st.title("👁️ Blind Assistant")

# Master Speech Data Holder
active_voice_instruction = ""
active_alert_instruction = ""

# Location
location = streamlit_geolocation()
u_lat = location.get('latitude') if location else None
u_lng = location.get('longitude') if location else None

col_v, col_c = st.columns([2, 1])

with col_v:
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col_c:
    st.subheader("Settings")
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", os.getenv("GOOGLE_MAPS_API_KEY", ""))
    dest_in = st.text_input("Destination", placeholder="e.g. Hope College")

    if st.button("🚀 START NAVIGATION", use_container_width=True):
        if u_lat and api_key and dest_in:
            data, err = get_walking_directions((u_lat, u_lng), dest_in, api_key)
            if data:
                st.session_state.update({"nav_steps": data, "nav_idx": 0, "nav_active": True})
                active_voice_instruction = "Navigation started. " + data[0]['text']
            else: st.error(err)

    if st.session_state.nav_active:
        steps = st.session_state.nav_steps
        idx = st.session_state.nav_idx
        if idx < len(steps):
            step = steps[idx]
            st.info(f"**Step {idx+1}:** {step['text']}")
            if step['text'] != st.session_state.last_nav:
                active_voice_instruction = step['text']
                st.session_state.last_nav = step['text']
                st.session_state.nav_idx += 1
        else:
            active_voice_instruction = "Arrived at destination."
            st.session_state.nav_active = False

    st.divider()
    if ctx.video_processor:
        with ctx.video_processor.lock:
            objs = ctx.video_processor.detections.copy()
        if objs:
            st.write(", ".join(objs))
            if "last_v_time" not in st.session_state: st.session_state.last_v_time = 0
            if time.time() - st.session_state.last_v_time > 5:
                active_alert_instruction = "I see " + " and ".join(objs[:2])
                st.session_state.last_v_time = time.time()

# ==========================================
# PERSISTENT VOICE ENGINE (Fixed Logic)
# ==========================================
# We use localStorage to remember the "Unlock" even if the iframe rerenders
voice_hub_json = json.dumps({"nav": active_voice_instruction, "alert": active_alert_instruction})

components.html(f"""
    <div style="background:#fef2f2; border:2px solid #ef4444; padding:15px; border-radius:12px; text-align:center;">
        <button id="ubtn" onclick="unlock()" style="background:#ef4444; color:white; border:none; padding:12px; border-radius:8px; cursor:pointer; font-weight:bold; width:100%; font-size:16px;">
            🔊 TAP TO RE-ENABLE VOICE 🚨
        </button>
        <div id="status" style="margin-top:8px; font-size:12px; color:#ef4444;">Voice status: Monitoring...</div>
    </div>
    <script>
    const data = {voice_hub_json};
    
    function unlock() {{
        localStorage.setItem('blind_voice_unlocked', 'true');
        updateUI();
        speak("Voice confirmed. Standing by.");
    }}

    function updateUI() {{
        if(localStorage.getItem('blind_voice_unlocked') === 'true') {{
            let b = document.getElementById('ubtn');
            b.style.background = "#10b981";
            b.innerText = "✔️ VOICE ACTIVE";
            document.getElementById('status').innerText = "AI Assistant Live";
            document.getElementById('status').style.color = "#10b981";
        }}
    }}

    function speak(t) {{
        if(localStorage.getItem('blind_voice_unlocked') !== 'true') return;
        window.speechSynthesis.cancel();
        let u = new SpeechSynthesisUtterance(t);
        u.rate = 1.0;
        window.speechSynthesis.speak(u);
    }}

    // Check memory and speak if needed
    updateUI();
    if (data.nav || data.alert) {{
        // Use a hash to avoid repeating the exact same message within the same iframe life
        const hash = data.nav + data.alert;
        if(window.lastH !== hash) {{
            speak(data.nav + " " + data.alert);
            window.lastH = hash;
        }}
    }}
    </script>
""", height=120)

time.sleep(1.5)
st.rerun()
