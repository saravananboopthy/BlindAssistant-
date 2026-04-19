import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_geolocation import streamlit_geolocation
import av
import cv2
import threading
import time
import os
import googlemaps
import re
import json
from ultralytics import YOLO
from dotenv import load_dotenv

# Load ENV
load_dotenv()

st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

# ==========================================
# CUSTOM STYLES
# ==========================================
st.markdown("""
<style>
.main-header { background: linear-gradient(135deg, #00d4aa 0%, #667eea 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem; }
.det-card { padding: 12px; border-radius: 10px; margin-bottom: 8px; background: rgba(0,212,170,0.1); border-left: 5px solid #00d4aa; font-weight: bold; }
.danger { background: rgba(255,107,107,0.1); border-left-color: #ff6b6b; color: #ff6b6b; }
.step-card { background: #eef2f7; padding: 15px; border-radius: 10px; border-left: 5px solid #667eea; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "nav_steps" not in st.session_state:
    st.session_state.update({"nav_steps": [], "nav_idx": 0, "engine_active": False})

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
        if not res: return None, "No path found."
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
    def recv(self, frame):
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
                cv2.rectangle(img, (x1,y1), (x2,y2), (0, 212, 170), 2)
            with self.lock: self.detections = detected
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI TOP SECTION
# ==========================================
st.markdown('<div class="main-header"><h1>👁️ Blind Assistant</h1></div>', unsafe_allow_html=True)
location = streamlit_geolocation()
u_lat = location.get('latitude') if location else None
u_lng = location.get('longitude') if location else None

# ==========================================
# FAIL-SAFE VOICE ENGINE (No Custom Component Folder Required)
# ==========================================
current_dets = []
col_v, col_i = st.columns([1.5, 1])

with col_v:
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=BlindProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )
    if ctx.video_processor:
        st.session_state.engine_active = st.toggle("Activate AI Voice Engine", value=st.session_state.engine_active)
        with ctx.video_processor.lock:
            current_dets = ctx.video_processor.detections.copy()

# State values for the JS Engine
nav_steps = st.session_state.nav_steps
nav_idx = st.session_state.nav_idx
engine_on = st.session_state.engine_active

# Robust JS Template
js_engine = f"""
    <div id="ui" style="background:#f1f5f9; padding:15px; border-radius:12px; text-align:center; font-family:sans-serif; border:2px solid #cbd5e1;">
        <button id="btn" onclick="unlock()" style="background:#ef4444; color:white; border:none; padding:12px; border-radius:8px; cursor:pointer; font-weight:bold; width:100%; font-size:16px;">
            🚨 TAP HERE TO UNLOCK VOICE 🔊
        </button>
        <div id="msg" style="margin-top:10px; font-size:13px; color:#64748b;">Voice status: Locked</div>
    </div>
    <script>
    const parent = window.parent;
    
    function speak(t, p=false) {{
        if (!window.unlocked) return;
        if (window.speechSynthesis.speaking && !p) return;
        if (p) window.speechSynthesis.cancel();
        let u = new SpeechSynthesisUtterance(t); u.rate = 1.0;
        window.speechSynthesis.speak(u);
    }}

    function unlock() {{
        window.unlocked = true;
        let btn = document.getElementById('btn');
        btn.style.background = '#10b981';
        btn.innerText = '✔️ VOICE ACTIVE';
        document.getElementById('msg').innerText = 'AI Assistant is monitoring...';
        speak("Assistant Live");
    }}

    // State from Python
    const dets = {json.dumps(current_dets)};
    const steps = {json.dumps(nav_steps)};
    let idx = {nav_idx};
    const active = {str(engine_on).lower()};
    const uLat = {u_lat if u_lat else 'null'};
    const uLng = {u_lng if u_lng else 'null'};

    // Persistence Check (Survives Reruns)
    const lastHash = sessionStorage.getItem('last_spoken_hash');
    const currentHash = JSON.stringify(dets) + JSON.stringify(steps) + idx + active;

    if (active && window.unlocked && lastHash !== currentHash) {{
        sessionStorage.setItem('last_spoken_hash', currentHash);
        
        // 1. Navigation Alert
        if (steps.length > 0 && idx < steps.length) {{
            const s = steps[idx];
            if (sessionStorage.getItem('last_nav') !== s.text) {{
                speak("Navigation Update: " + s.text, true);
                sessionStorage.setItem('last_nav', s.text);
            }}
            
            // Auto-Progress Logic (JavaScript side)
            if (uLat && uLng) {{
                const R = 6371e3;
                const dLat = (s.lat - uLat) * Math.PI/180;
                const dLng = (s.lng - uLng) * Math.PI/180;
                const a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(uLat * Math.PI/180) * Math.cos(s.lat * Math.PI/180) * Math.sin(dLng/2) * Math.sin(dLng/2);
                const dist = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                
                if (dist < 15) {{
                    parent.postMessage({{type: 'streamlit:set_query_params', queryParams: {{'auto_next': Date.now()}} }}, '*');
                }}
            }}
        }}

        // 2. Object Logic (Throttled)
        if (dets.length > 0) {{
            let danger = false;
            let parts = [];
            let seen = new Set();
            for (let d of dets) {{
                if (!seen.has(d.label)) {{
                    parts.push(d.label + " " + d.pos);
                    seen.add(d.label);
                    if (d.dist === 'near') danger = true;
                }}
                if (parts.length >= 2) break;
            }}
            const msgBody = (danger ? "Watch out! " : "Ahead is ") + parts.join(" and ");
            const now = Date.now();
            const lastDet = sessionStorage.getItem('last_det');
            const lastDetT = parseInt(sessionStorage.getItem('last_det_t') || "0");
            
            if (lastDet !== msgBody || (now - lastDetT > 10000)) {{
                speak(msgBody, danger);
                sessionStorage.setItem('last_det', msgBody);
                sessionStorage.setItem('last_det_t', now.toString());
            }}
        }}
    }}
    </script>
"""
components.html(js_engine, height=130)

# ==========================================
# AUTO-NEXT LISTENER
# ==========================================
if st.query_params.get("auto_next"):
    if st.session_state.nav_idx < len(st.session_state.nav_steps) - 1:
        st.session_state.nav_idx += 1
        st.query_params.clear()
        st.rerun()

with col_i:
    st.subheader("📍 Sensors")
    if u_lat: st.success(f"GPS Locked: {u_lat:.4f}, {u_lng:.4f}")
    else: st.warning("Waiting for GPS...")

    default_api = os.getenv("GOOGLE_MAPS_API_KEY", "")
    try:
        if st.secrets.get("GOOGLE_MAPS_API_KEY"): default_api = st.secrets["GOOGLE_MAPS_API_KEY"]
    except: pass
    
    api = st.text_input("G-Maps Key", value=default_api, type="password")
    dest_in = st.text_input("Destination", placeholder="e.g. Hope College, Coimbatore")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Navigation", type="primary", use_container_width=True):
            if u_lat and api and dest_in:
                steps, err = get_walking_directions(f"{u_lat},{u_lng}", dest_in, api)
                if steps:
                    st.session_state.update({"nav_steps": steps, "nav_idx": 0})
                else: st.error(err)
    with c2:
        if st.button("Clear Route", use_container_width=True):
            st.session_state.nav_steps = []; st.session_state.nav_idx = 0; st.rerun()

    if st.session_state.nav_steps:
        i = st.session_state.nav_idx
        st.markdown(f'<div class="step-card"><b>Step {i+1}:</b><br>{st.session_state.nav_steps[i]["text"]}</div>', unsafe_allow_html=True)

    st.subheader("🎯 Objects")
    for d in current_dets:
        st.markdown(f'<div class="det-card {"danger" if d["dist"]=="near" else ""}">{d["label"].upper()} {d["pos"]}</div>', unsafe_allow_html=True)

if st.session_state.engine_active:
    time.sleep(1)
    st.rerun()
