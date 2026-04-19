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
# PAGE SETTINGS
# ==========================================
st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

st.markdown("""
<style>
.main-header { background: #1e293b; padding: 15px; border-radius: 12px; color: white; text-align: center; }
.stButton>button { border-radius: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Session State for Continuous Tracking
if "state" not in st.session_state:
    st.session_state.state = {
        "nav_steps": [], "nav_idx": 0, "active": False, 
        "last_nav_msg": "", "obj_memory": {}, "run_camera": True
    }

# ==========================================
# VISION PROCESSOR
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
# HELPER FUNCTIONS
# ==========================================
def get_walking_directions(source, dest, api_key):
    try:
        gmaps = googlemaps.Client(key=api_key)
        res = gmaps.directions(source, dest, mode="walking")
        if not res: return None, "No route found."
        leg = res[0]["legs"][0]
        steps = []
        for s in leg["steps"]:
            instr = re.sub(r"<.*?>", "", s["html_instructions"]).replace("&nbsp;", " ").strip()
            steps.append({"text": instr, "lat": s["end_location"]["lat"], "lng": s["end_location"]["lng"]})
        return steps, None
    except Exception as e: return None, str(e)

def calculate_dist(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2); dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ==========================================
# UI & TRACKING
# ==========================================
st.markdown('<div class="main-header"><h1>👁️ Blind Assistant</h1></div>', unsafe_allow_html=True)

# Data for JS Voice
nav_instruction = ""
alert_instruction = ""

# GPS
location = streamlit_geolocation()
u_lat = location.get('latitude') if location else None
u_lng = location.get('longitude') if location else None

col_v, col_c = st.columns([1.5, 1])

with col_v:
    if st.session_state.state["run_camera"]:
        ctx = webrtc_streamer(
            key="v_stream",
            video_processor_factory=VisionProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False}
        )
    else:
        st.info("System Offline. Click Activate to restart.")

with col_c:
    st.subheader("Navigation & Control")
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", os.getenv("GOOGLE_MAPS_API_KEY", ""))
    dest_in = st.text_input("Destination")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 ACTIVATE", use_container_width=True):
            if u_lat and api_key and dest_in:
                data, err = get_walking_directions((u_lat, u_lng), dest_in, api_key)
                if data:
                    st.session_state.state.update({"nav_steps": data, "nav_idx": 0, "active": True, "run_camera": True})
                    nav_instruction = "Navigation started. " + data[0]['text']
                else: st.error(err)
    with c2:
        if st.button("🛑 STOP", type="primary", use_container_width=True):
            st.session_state.state.update({"active": False, "run_camera": False, "nav_steps": []})
            st.rerun()

    # LIVE GPS PROGRESSION LOGIC
    if st.session_state.state["active"] and u_lat:
        steps = st.session_state.state["nav_steps"]
        idx = st.session_state.state["nav_idx"]
        
        if idx < len(steps):
            cur_step = steps[idx]
            dist = calculate_dist(u_lat, u_lng, cur_step['lat'], cur_step['lng'])
            
            st.success(f"**Target:** {cur_step['text']}")
            st.metric("Distance to Turn", f"{dist:.1f} m")
            
            # If we are within 15m, advance to next step AND speak it
            if dist < 15:
                # Speak new instruction only once
                if cur_step['text'] != st.session_state.state["last_nav_msg"]:
                    nav_instruction = "Next: " + cur_step['text']
                    st.session_state.state["last_nav_msg"] = cur_step['text']
                
                if st.session_state.state["nav_idx"] < len(steps) - 1:
                    st.session_state.state["nav_idx"] += 1
        else:
            nav_instruction = "Arrival."
            st.session_state.state["active"] = False

    st.divider()
    if st.session_state.state["run_camera"] and ctx.video_processor:
        with ctx.video_processor.lock:
            objs = ctx.video_processor.detections.copy()
        if objs:
            st.write(", ".join(objs))
            now = time.time()
            for o in objs:
                # 15 second memory to prevent repetition
                if o not in st.session_state.state["obj_memory"] or now - st.session_state.state["obj_memory"][o] > 15:
                    alert_instruction = "Ahead: " + " and ".join(objs[:2])
                    st.session_state.state["obj_memory"][o] = now
                    break

# ==========================================
# MASTER PERSISTENT VOICE ENGINE
# ==========================================
voice_hub = json.dumps({"nav": nav_instruction, "alert": alert_instruction})

components.html(f"""
    <div style="background:#fef2f2; border:1px solid #ef4444; padding:10px; border-radius:10px; text-align:center;">
        <button id="vbtn" onclick="lck()" style="background:#ef4444; color:white; border:none; padding:8px; border-radius:5px; cursor:pointer; font-weight:bold; width:100%;">
            🔊 TAP TO RE-ENABLE VOICE 🚨
        </button>
    </div>
    <script>
    const vdata = {voice_hub};
    function lck() {{ localStorage.setItem('v_unlocked', 'true'); update(); speak("Assistant Ready."); }}
    function update() {{ if(localStorage.getItem('v_unlocked') === 'true') {{ let b = document.getElementById('vbtn'); b.style.background = "#10b981"; b.innerText = "✔️ VOICE ACTIVE"; }} }}
    function speak(t) {{
        if(localStorage.getItem('v_unlocked') !== 'true') return;
        window.speechSynthesis.cancel();
        let u = new SpeechSynthesisUtterance(t); u.rate = 1.0;
        window.speechSynthesis.speak(u);
    }}
    update();
    if (vdata.nav || vdata.alert) {{
        const h = vdata.nav + vdata.alert;
        if(window.lastH !== h) {{ speak(vdata.nav + " " + vdata.alert); window.lastH = h; }}
    }}
    </script>
""", height=80)

if st.session_state.state["active"] or st.session_state.state["run_camera"]:
    time.sleep(1.5)
    st.rerun()
