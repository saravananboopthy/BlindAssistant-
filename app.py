import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_geolocation import streamlit_geolocation
import av
import cv2
import threading
import time
import math
import os
import googlemaps
import re
from collections import Counter
from ultralytics import YOLO

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

# Session State Initialization
if "nav_steps" not in st.session_state:
    st.session_state.update({
        "nav_steps": [], 
        "nav_idx": 0, 
        "nav_active": False, 
        "last_nav": "",
        "detected_memory": {}
    })

# ==========================================
# SPEECH ENGINE
# ==========================================
def speak(text):
    # Using a unique key (timestamp) ensures Streamlit creates a fresh iframe that triggers the script
    components.html(f"""
        <script>
        window.speechSynthesis.cancel(); // Stop any current speech
        const msg = new SpeechSynthesisUtterance("{text}");
        msg.rate = 1.0;
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0, key=f"speak_{time.time()}")

# ==========================================
# NAVIGATION LOGIC
# ==========================================
def get_walking_directions(source, dest, api_key):
    if not api_key: return None, "Missing API Key"
    try:
        gmaps = googlemaps.Client(key=api_key)
        res = gmaps.directions(source, dest, mode="walking")
        if not res: return None, "No walking route found."
        
        leg = res[0]["legs"][0]
        steps = []
        for s in leg["steps"]:
            instr = re.sub(r"<.*?>", "", s["html_instructions"]).replace("&nbsp;", " ").strip()
            steps.append({
                "text": f"{instr} for {s['distance']['text']}",
                "lat": s["end_location"]["lat"],
                "lng": s["end_location"]["lng"]
            })
        return steps, None
    except Exception as e: return None, str(e)

def distance_meters(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ==========================================
# VISION PROCESSOR
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class BlindProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = {}
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = model(img, conf=0.45, verbose=False)[0]
        detected = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            detected.append(label)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        with self.lock: self.detections = dict(Counter(detected))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI LAYOUT
# ==========================================
st.title("👁️ Blind Assistant")

# Auto-loading GPS
location = streamlit_geolocation()
u_lat = location.get('latitude') if location else None
u_lng = location.get('longitude') if location else None

col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="camera_stream",
        video_processor_factory=BlindProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("🧭 Path Input")
    
    # Secret Key auto-load
    default_api = os.getenv("GOOGLE_MAPS_API_KEY", "")
    try:
        if st.secrets.get("GOOGLE_MAPS_API_KEY"): default_api = st.secrets["GOOGLE_MAPS_API_KEY"]
    except: pass
    
    api = st.text_input("G-Maps Key", value=default_api, type="password")
    dest = st.text_input("Destination", placeholder="e.g. Hope College, Coimbatore")

    if st.button("Start Navigation", type="primary"):
        if u_lat and api and dest:
            steps, err = get_walking_directions((u_lat, u_lng), dest, api)
            if steps:
                st.session_state.nav_steps = steps
                st.session_state.nav_idx = 0
                st.session_state.nav_active = True
                speak("Navigation started. Path found.")
            else: st.error(err)

    if st.session_state.nav_active:
        steps = st.session_state.nav_steps
        idx = st.session_state.nav_idx
        if idx < len(steps):
            step = steps[idx]
            st.info(f"**Step {idx+1}:** {step['text']}")
            
            # Auto-Voice Navigation Logic
            if u_lat:
                dist = distance_meters(u_lat, u_lng, step['lat'], step['lng'])
                if dist <= 15: # Proximity trigger (15 meters)
                    if step['text'] != st.session_state.last_nav:
                        speak(step['text'])
                        st.session_state.last_nav = step['text']
                        st.session_state.nav_idx += 1
        else:
            st.success("Arrived at Destination")
            speak("You have arrived at your destination.")
            st.session_state.nav_active = False

    st.divider()
    st.subheader("🎯 Objects Nearby")
    if ctx.video_processor:
        with ctx.video_processor.lock:
            detections = ctx.video_processor.detections.copy()
        
        if detections:
            for obj, count in detections.items():
                st.markdown(f"**{obj.upper()}**: Detected")
                
                # Speech logic with 3-second throttle
                now = time.time()
                if obj not in st.session_state.detected_memory or now - st.session_state.detected_memory[obj] > 4:
                    speak(obj)
                    st.session_state.detected_memory[obj] = now
        else:
            st.write("Scanning for obstacles...")

# Controlled refresh loop
if ctx.state.playing:
    time.sleep(1)
    st.rerun()
