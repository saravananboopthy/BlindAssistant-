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
</style>
""", unsafe_allow_html=True)

# ==========================================
# JS GEOLOCATION (High Accuracy)
# ==========================================
# This script sends lat/lng to Streamlit query params so we can read it
components.html("""
    <script>
    function sendLoc() {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const lat = pos.coords.latitude;
                const lng = pos.coords.longitude;
                window.parent.postMessage({
                    type: 'streamlit:set_query_params',
                    queryParams: { lat: lat, lng: lng }
                }, '*');
            },
            (err) => { console.error("Loc Error:", err); },
            { enableHighAccuracy: true }
        );
    }
    sendLoc();
    setInterval(sendLoc, 10000); // Update every 10s
    </script>
""", height=0)

user_lat = st.query_params.get("lat")
user_lng = st.query_params.get("lng")

# ==========================================
# UTILS & MODEL
# ==========================================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

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
            dist = "near" if dist_px > 300 else "far"
            
            detected_now.append({"label": label, "pos": pos, "dist": dist})

            # Draw
            color = (139, 68, 255) if dist == "near" else (0, 212, 170)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {dist}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with self.lock:
            self.detections = detected_now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# SESSION STATE
# ==========================================
if "nav_steps" not in st.session_state:
    st.session_state.update({
        "nav_steps": [], "nav_idx": 0, "destination": "", 
        "last_spoken": "", "speak_trigger": 0
    })

# ==========================================
# JS TTS HELPER
# ==========================================
def browser_speak(text, rate=0.7):
    if not text: return
    clean_text = text.replace("'", "").replace('"', "")
    # Use unique key to trigger refresh of speech component
    st.session_state.speak_trigger += 1
    components.html(f"""
        <script>
            window.speechSynthesis.cancel();
            var u = new SpeechSynthesisUtterance("{clean_text}");
            u.rate = {rate};
            window.speechSynthesis.speak(u);
        </script>
    """, height=0, key=f"tts_{st.session_state.speak_trigger}")

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown(f"""
<div class="main-header">
    <h1>👁️ Blind Assistant - Cloud Edition</h1>
    <p>GPS Lat: {user_lat or 'Finding...'} | Lng: {user_lng or 'Finding...'}</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Controls")
    api_key = st.text_input("Google Maps API Key", value=os.getenv("GOOGLE_MAPS_API_KEY", ""), type="password")
    
    st.divider()
    dest_input = st.text_input("📍 Set Destination", placeholder="e.g. Coimbatore Bus Stand")
    if st.button("Start Navigation") and dest_input and api_key:
        if not user_lat or not user_lng:
            st.error("Waiting for GPS location from browser. Please allow location access.")
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
                    st.success(f"Route found to {dest_input}")
                    browser_speak(f"Navigating to {dest_input}. Your first step is: {steps[0]}")
                else:
                    st.error("No walking route found.")
            except Exception as e:
                st.error(f"Google Maps Error: {e}")

col_vid, col_info = st.columns([3, 2])

with col_vid:
    webrtc_ctx = webrtc_streamer(
        key="vision",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col_info:
    st.subheader("🧭 Navigation")
    if st.session_state.nav_steps:
        idx = st.session_state.nav_idx
        step_text = st.session_state.nav_steps[idx]
        st.markdown(f"""<div class="nav-card"><h3>Step {idx+1}/{len(st.session_state.nav_steps)}</h3><p style="font-size:1.2rem">{step_text}</p></div>""", unsafe_allow_html=True)
        if st.button("Next Step ➡"):
            if st.session_state.nav_idx < len(st.session_state.nav_steps) - 1:
                st.session_state.nav_idx += 1
                new_step = st.session_state.nav_steps[st.session_state.nav_idx]
                browser_speak(new_step)
            else:
                browser_speak("You have arrived at your destination.")
    else:
        st.info("Enter a destination and click 'Start' to begin GPS navigation.")

    st.divider()
    st.subheader("🎯 Active Detections")
    det_placeholder = st.empty()

# ==========================================
# MAIN LOOP (TTS & UPDATE)
# ==========================================
if webrtc_ctx.video_processor:
    while True:
        with webrtc_ctx.video_processor.lock:
            current_detections = webrtc_ctx.video_processor.detections.copy()
        
        with det_placeholder.container():
            if current_detections:
                for d in current_detections:
                    is_near = d['dist'] == "near"
                    css = "warn" if is_near else ""
                    st.markdown(f"""<div class="det-item {css}"><b>{d['label'].upper()}</b> {d['pos']}</div>""", unsafe_allow_html=True)
                
                # TTS Logic: Speak the top item
                top_obj = current_detections[0]
                speak_msg = f"{top_obj['label']} {top_obj['pos']}"
                
                if speak_msg != st.session_state.last_spoken:
                    st.session_state.last_spoken = speak_msg
                    # SLOW VOICE as requested
                    browser_speak(speak_msg, rate=0.7)
            else:
                st.write("Checking path...")
        
        time.sleep(2.0) # Line by line spacing