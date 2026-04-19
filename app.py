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

# Load environment variables
load_dotenv()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Blind Assistant | Cloud Ready", 
    page_icon="👁️", 
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.det-item { 
    padding: 12px; border-radius: 10px; margin-bottom: 8px; 
    background: rgba(0,212,170,0.08); border-left: 5px solid #00d4aa; 
    font-weight: 500;
}
.warn { background: rgba(255,107,107,0.08); border-left-color: #ff6b6b; }
.main-header {
    background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(102,126,234,0.1));
    padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "nav_steps" not in st.session_state:
    st.session_state.update({
        "nav_steps": [], "nav_idx": 0, "destination": "", 
        "last_spoken": "", "speak_now": "", "engine_active": False
    })

# ==========================================
# HELPER COMPONENTS (Geo + Voice + TTS)
# ==========================================
# We use a single persistent JS component to manage browser APIs
msg_to_speak = st.session_state.get("speak_now", "")

components.html(f"""
    <div style="display: flex; gap: 12px; align-items: center; padding: 10px; background: #f8f9fa; border-radius: 30px;">
        <button id="mic-btn" style="background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; font-weight: 600; outline: none;">
            🎤 Speak Destination
        </button>
        <span id="status" style="color: #666; font-size: 0.9rem;">Ready</span>
    </div>

    <script>
    // 1. HIGH-PRECISION GEOLOCATION
    function sendLoc() {{
        navigator.geolocation.getCurrentPosition((pos) => {{
            const lat = pos.coords.latitude;
            const lng = pos.coords.longitude;
            window.parent.postMessage({{ 
                type: 'streamlit:set_query_params', 
                queryParams: {{ lat: lat.toFixed(6), lng: lng.toFixed(6) }} 
            }}, '*');
        }}, (err) => {{ 
            console.error("GPS Error:", err.message);
        }}, {{ enableHighAccuracy: true, timeout: 8000, maximumAge: 0 }});
    }}
    setInterval(sendLoc, 8000); 
    sendLoc();

    // 2. WEB SPEECH FOR DESTINATION
    const micBtn = document.getElementById('mic-btn');
    const status = document.getElementById('status');
    
    if ('webkitSpeechRecognition' in window) {{
        const recognition = new webkitSpeechRecognition();
        micBtn.onclick = () => {{ 
            recognition.start(); 
            status.innerText = "Listening..."; 
            micBtn.style.background = "#ff6b6b";
        }};
        recognition.onresult = (e) => {{
            const text = e.results[0][0].transcript;
            status.innerText = "Found: " + text;
            micBtn.style.background = "#667eea";
            window.parent.postMessage({{ type: 'streamlit:set_query_params', queryParams: {{ dest: text }} }}, '*');
        }};
    }}

    // 3. SECURE TEXT-TO-SPEECH (No Cut-off)
    var msg = "{msg_to_speak}";
    if (msg && window.lastPlayedMsg !== msg) {{
        var u = new SpeechSynthesisUtterance(msg);
        u.rate = 0.8;
        window.speechSynthesis.speak(u);
        window.lastPlayedMsg = msg;
    }}
    </script>
""", height=70)

# Sync parameters from URL to Session State
user_lat = st.query_params.get("lat")
user_lng = st.query_params.get("lng")
if st.query_params.get("dest"):
    st.session_state.destination = st.query_params.get("dest")

# ==========================================
# AI MODEL LOADING
# ==========================================
@st.cache_resource
def load_yolo():
    try:
        # Reverted to Nano (fastest) to ensure no lag in cloud environment
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"AI Engine Initialization Failed: {e}")
        return None

model = load_yolo()

def browser_speak(text):
    if text and text != st.session_state.last_spoken:
        st.session_state.speak_now = text
        st.session_state.last_spoken = text
        # Faster trigger for voice feedback
        time.sleep(0.2)
        st.rerun()

# ==========================================
# VISION ENGINE (WEBRTC)
# ==========================================
class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if model:
            results = model(img, conf=0.4, verbose=False)[0]
            detected_now = []
            
            h, w, _ = img.shape
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                
                # Spatial positioning
                x_center = (x1 + x2) / 2
                if x_center < w * 0.33: pos = "on your left"
                elif x_center > w * 0.66: pos = "on your right"
                else: pos = "ahead"

                # Distance estimation (based on bounding box width)
                dist = "near" if (x2 - x1) > 280 else "far"
                detected_now.append({"label": label, "pos": pos, "dist": dist})

                # Visual tags
                color = (139, 68, 255) if dist == "near" else (0, 212, 170)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            with self.lock:
                self.detections = detected_now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown(f"""
<div class="main-header">
    <h1 style="margin:0; color:#1e293b;">👁️ Blind Assistant</h1>
    <p style="margin:5px 0 0 0; color:#64748b;">
        <b>GPS:</b> {'Active' if user_lat else 'Searching...'} | 
        <b>Lat:</b> {user_lat or '---'} | <b>Lng:</b> {user_lng or '---'}
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("G-Maps API Key", value=os.getenv("GOOGLE_MAPS_API_KEY", ""), type="password")
    
    st.divider()
    dest_input = st.text_input("📍 Destination", value=st.session_state.destination, placeholder="e.g. Hospital")
    
    if st.button("🚀 Start Navigation", use_container_width=True):
        if not user_lat or not user_lng:
            st.warning("Waiting for GPS signal...")
        elif not api_key:
            st.error("Missing Google Maps API Key")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                res = gmaps.directions((float(user_lat), float(user_lng)), dest_input, mode="walking")
                if res:
                    leg = res[0]["legs"][0]
                    steps = [s["html_instructions"].replace("<b>", "").replace("</b>", "").replace('<div style="font-size:0.9em">', " ").replace("</div>", "") for s in leg["steps"]]
                    st.session_state.update({"nav_steps": steps, "nav_idx": 0, "destination": dest_input})
                    browser_speak(f"Navigating to {dest_input}. First step: {steps[0]}")
                else:
                    st.error("No route found.")
            except Exception as e:
                st.error(f"Navigation Error: {e}")

    st.divider()
    if st.button("🛑 STOP ALL", type="primary", use_container_width=True):
        st.session_state.engine_active = False
        st.session_state.nav_steps = []
        st.session_state.speak_now = ""
        st.rerun()

col_vid, col_info = st.columns([1.8, 1])

with col_vid:
    webrtc_ctx = webrtc_streamer(
        key="vision",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    if webrtc_ctx.video_processor:
        st.session_state.engine_active = st.toggle("Activate AI Voice Engine", value=st.session_state.engine_active)

with col_info:
    st.subheader("🧭 Path Guidance")
    if st.session_state.nav_steps:
        idx = st.session_state.nav_idx
        step_text = st.session_state.nav_steps[idx]
        st.success(f"Step {idx+1}/{len(st.session_state.nav_steps)}")
        st.markdown(f"**{step_text}**")
        if st.button("Next Step ➡"):
            if st.session_state.nav_idx < len(st.session_state.nav_steps) - 1:
                st.session_state.nav_idx += 1
                browser_speak(st.session_state.nav_steps[st.session_state.nav_idx])
            else:
                browser_speak("Goal reached.")
    else:
        st.info("Input a destination to begin guidance.")

    st.divider()
    st.subheader("🎯 Objects Nearby")
    det_placeholder = st.empty()

# ==========================================
# REFRESH / TTS LOOP
# ==========================================
if webrtc_ctx.video_processor and st.session_state.engine_active:
    with webrtc_ctx.video_processor.lock:
        current_detections = webrtc_ctx.video_processor.detections.copy()
    
    with det_placeholder.container():
        if current_detections:
            for d in current_detections:
                is_near = d['dist'] == "near"
                css = "warn" if is_near else ""
                st.markdown(f"""<div class="det-item {css}">{d['label'].upper()} {d['pos']}</div>""", unsafe_allow_html=True)
            
            # Prioritize near objects for speech
            nears = [d for d in current_detections if d['dist'] == "near"]
            top = nears[0] if nears else current_detections[0]
            msg = f"Watch out, {top['label']} {top['pos']}" if top['dist'] == "near" else f"{top['label']} {top['pos']}"
            
            if msg != st.session_state.last_spoken:
                browser_speak(msg)
        else:
            st.write("Checking...")

    # Faster refresh for near-instant response
    time.sleep(0.5)
    st.rerun()
