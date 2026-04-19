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
import math

# Load environment variables
load_dotenv()

# ==========================================
# PAGE CONFIG & CSS
# ==========================================
st.set_page_config(
    page_title="Blind Assistant | Cloud Edition",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(102,126,234,0.08));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.main-header h1 {
    background: linear-gradient(135deg, #00d4aa, #667eea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem;
}
.nav-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(0,212,170,0.05));
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0;
}
.det-item {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.6rem 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;
    background: rgba(0,212,170,0.06); border-left: 3px solid #00d4aa;
}
.det-item.warn { background: rgba(255,107,107,0.06); border-left-color: #ff6b6b; }
.stButton>button { width: 100%; border-radius: 8px; transition: all 0.3s; }
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,212,170,0.2); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "nav_steps" not in st.session_state:
    st.session_state.update({
        "nav_steps": [], "nav_idx": 0, "destination": "", 
        "last_spoken": "", "speak_now": "", "speak_trigger": 0,
        "engine_active": False
    })

# ==========================================
# JS HELPER COMPONENT (Geo + Voice Input + TTS)
# ==========================================
# This component handles 3 things:
# 1. Periodically sends location to query params
# 2. Provides a Voice Input button for destination
# 3. Listens for the 'speak_now' state and speaks it
msg_to_speak = st.session_state.get("speak_now", "")

components.html(f"""
    <div style="display: flex; gap: 10px; align-items: center; font-family: sans-serif;">
        <button id="mic-btn" style="background: #667eea; color: white; border: none; padding: 8px 15px; border-radius: 20px; cursor: pointer; font-weight: 600;">
            🎤 Speak Destination
        </button>
        <span id="status" style="font-size: 0.8rem; color: #666;">Ready</span>
    </div>

    <script>
    // 1. GEOLOCATION
    function sendLoc() {{
        navigator.geolocation.getCurrentPosition(
            (pos) => {{
                const lat = pos.coords.latitude;
                const lng = pos.coords.longitude;
                const params = new URLSearchParams(window.parent.location.search);
                if (params.get('lat') != lat || params.get('lng') != lng) {{
                    window.parent.postMessage({{
                        type: 'streamlit:set_query_params',
                        queryParams: {{ lat: lat, lng: lng }}
                    }}, '*');
                }}
            }},
            (err) => {{ console.error("Loc Error:", err); }},
            {{ enableHighAccuracy: true }}
        );
    }}
    setInterval(sendLoc, 10000);
    sendLoc();

    // 2. VOICE INPUT
    const micBtn = document.getElementById('mic-btn');
    const status = document.getElementById('status');
    
    if ('webkitSpeechRecognition' in window) {{
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        micBtn.onclick = () => {{
            recognition.start();
            status.innerText = "Listening...";
        }};

        recognition.onresult = (event) => {{
            const text = event.results[0][0].transcript;
            status.innerText = "Found: " + text;
            window.parent.postMessage({{
                type: 'streamlit:set_query_params',
                queryParams: {{ 
                    lat: new URLSearchParams(window.parent.location.search).get('lat'), 
                    lng: new URLSearchParams(window.parent.location.search).get('lng'),
                    dest: text 
                }}
            }}, '*');
        }};
        
        recognition.onerror = () => {{ status.innerText = "Error!"; }};
    }} else {{
        micBtn.style.display = 'none';
        status.innerText = "Voice not supported";
    }}

    // 3. TTS (VOICE OUTPUT)
    var msg = "{msg_to_speak}";
    if (msg && window.parent.lastSpoken !== msg) {{
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance(msg);
        u.rate = 0.7;
        window.speechSynthesis.speak(u);
        window.parent.lastSpoken = msg;
    }}
    </script>
""", height=50)

# Sync Voice Input to Session State
voice_dest = st.query_params.get("dest")
if voice_dest and voice_dest != st.session_state.destination:
    st.session_state.destination = voice_dest

user_lat = st.query_params.get("lat")
user_lng = st.query_params.get("lng")

# ==========================================
# UTILS & MODEL
# ==========================================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

def browser_speak(text):
    if text and text != st.session_state.last_spoken:
        st.session_state.speak_now = text
        st.session_state.last_spoken = text
        st.rerun()

# ==========================================
# VIDEO PROCESSOR (AI ENGINE)
# ==========================================
class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.35, verbose=False)[0]
        
        h, w, _ = img.shape
        detected_now = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            
            x_center = (x1 + x2) / 2
            if x_center < w * 0.33: pos = "on your left"
            elif x_center > w * 0.66: pos = "on your right"
            else: pos = "immediately ahead"

            dist_px = x2 - x1
            dist = "near" if dist_px > 250 else "far"
            
            detected_now.append({"label": label, "pos": pos, "dist": dist})

            # Draw UI
            color = (139, 68, 255) if dist == "near" else (0, 212, 170)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {dist}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with self.lock:
            self.detections = detected_now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown(f"""
<div class="main-header">
    <h1>👁️ Blind Assistant - Cloud Edition</h1>
    <p><b>Status:</b> {'OK' if user_lat else 'Waiting for Location...'} | <b>Lat:</b> {user_lat or '---'} | <b>Lng:</b> {user_lng or '---'}</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Controls")
    api_key = st.text_input("Google Maps API Key", value=os.getenv("GOOGLE_MAPS_API_KEY", ""), type="password")
    
    st.divider()
    dest_input = st.text_input("📍 Destination", value=st.session_state.destination, placeholder="Where to go?")
    
    if st.button("🚀 Start Navigation") and dest_input and api_key:
        if not user_lat or not user_lng:
            st.error("Enable location access and wait for coordinates.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                source_coords = (float(user_lat), float(user_lng))
                res = gmaps.directions(source_coords, dest_input, mode="walking")
                if res:
                    leg = res[0]["legs"][0]
                    steps = []
                    for s in leg["steps"]:
                        instr = s["html_instructions"].replace("<b>", "").replace("</b>", "").replace('<div style="font-size:0.9em">', " ").replace("</div>", "")
                        steps.append(instr)
                    st.session_state.nav_steps = steps
                    st.session_state.nav_idx = 0
                    st.session_state.destination = dest_input
                    st.success("Route Found!")
                    browser_speak(f"Navigating to {dest_input}. First step: {steps[0]}")
                else:
                    st.error("No walking route found.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    if st.button("🛑 STOP ALL", type="primary"):
        st.session_state.engine_active = False
        st.session_state.nav_steps = []
        st.session_state.speak_now = ""
        st.rerun()

col_vid, col_info = st.columns([3, 2])

with col_vid:
    webrtc_ctx = webrtc_streamer(
        key="vision",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    if webrtc_ctx.video_processor:
        if st.checkbox("Enable AI Voice Feedback", value=st.session_state.engine_active):
            st.session_state.engine_active = True
        else:
            st.session_state.engine_active = False

with col_info:
    st.subheader("🧭 Directions")
    if st.session_state.nav_steps:
        idx = st.session_state.nav_idx
        step_text = st.session_state.nav_steps[idx]
        st.markdown(f"""<div class="nav-card"><h3>Step {idx+1}/{len(st.session_state.nav_steps)}</h3><p style="font-size:1.1rem">{step_text}</p></div>""", unsafe_allow_html=True)
        if st.button("Next Step ➡"):
            if st.session_state.nav_idx < len(st.session_state.nav_steps) - 1:
                st.session_state.nav_idx += 1
                browser_speak(st.session_state.nav_steps[st.session_state.nav_idx])
            else:
                browser_speak("You have arrived at your destination.")
    else:
        st.info("Set destination in sidebar to start navigation.")

    st.divider()
    st.subheader("🎯 Active Detections")
    det_placeholder = st.empty()

# ==========================================
# MAIN LOOP (Polled Updates)
# ==========================================
if webrtc_ctx.video_processor and st.session_state.engine_active:
    with webrtc_ctx.video_processor.lock:
        current_detections = webrtc_ctx.video_processor.detections.copy()
    
    with det_placeholder.container():
        if current_detections:
            for d in current_detections:
                is_near = d['dist'] == "near"
                css = "warn" if is_near else ""
                st.markdown(f"""<div class="det-item {css}"><b>{d['label'].upper()}</b> {d['pos']}</div>""", unsafe_allow_html=True)
            
            # Speak only if something is NEAR or it's a new top-level object
            # Priority: Near objects
            near_objs = [d for d in current_detections if d['dist'] == "near"]
            top_obj = near_objs[0] if near_objs else current_detections[0]
            
            speak_msg = f"Watch out, {top_obj['label']} {top_obj['pos']}" if top_obj['dist'] == "near" else f"{top_obj['label']} {top_obj['pos']}"
            
            if speak_msg != st.session_state.last_spoken:
                st.session_state.speak_now = speak_msg
                st.session_state.last_spoken = speak_msg
                st.rerun() 
        else:
            st.write("Path clear...")
    
    time.sleep(2.0)
    st.rerun()
