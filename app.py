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
from collections import Counter
from ultralytics import YOLO

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

# Refresh the UI every second to check GPS/Objects (without restarting camera)
st_autorefresh(interval=1000, key="nav_refresh")

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
    # Unique ID as a string ensures no typing errors on the Cloud
    unique_id = str(int(time.time() * 1000))
    components.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.rate = 1.0;
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0, key=f"speak_{unique_id}")

# ==========================================
# NAVIGATION LOGIC
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
            steps.append({
                "text": f"{instr} for {s['distance']['text']}",
                "lat": s["end_location"]["lat"],
                "lng": s["end_location"]["lng"]
            })
        return steps, None
    except Exception as e: return None, str(e)

def calculate_dist(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ==========================================
# VISION PROCESSOR
# ==========================================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = {}
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = model(img, conf=0.45, verbose=False)[0]
        detected = [model.names[int(b.cls[0])] for b in res.boxes]
        with self.lock: self.detections = dict(Counter(detected))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI
# ==========================================
st.title("👁️ Blind Assistant")

# GPS TRACKER
location = streamlit_geolocation()
u_lat = location.get('latitude') if location else None
u_lng = location.get('longitude') if location else None

col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VisionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("🧭 Navigation")
    if u_lat: st.success(f"GPS Connected")
    
    # Auto-load API from secrets
    default_api = os.getenv("GOOGLE_MAPS_API_KEY", "")
    try:
        if st.secrets.get("GOOGLE_MAPS_API_KEY"): default_api = st.secrets["GOOGLE_MAPS_API_KEY"]
    except: pass
    
    api_key = st.text_input("G-Maps Key", value=default_api, type="password")
    dest = st.text_input("Destination", placeholder="e.g. Hope College")

    if st.button("Start Navigation", type="primary"):
        if u_lat and api_key and dest:
            data, err = get_walking_directions((u_lat, u_lng), dest, api_key)
            if data:
                st.session_state.nav_steps = data
                st.session_state.nav_idx = 0
                st.session_state.nav_active = True
                speak("Navigation started")
            else: st.error(err)

    if st.session_state.nav_active:
        steps = st.session_state.nav_steps
        curr = st.session_state.nav_idx
        if curr < len(steps):
            step = steps[curr]
            st.info(f"Step {curr+1}: {step['text']}")
            # Auto-Nav speech logic
            if u_lat:
                dist = calculate_dist(u_lat, u_lng, step['lat'], step['lng'])
                if dist <= 20 and step['text'] != st.session_state.last_nav:
                    speak(step['text'])
                    st.session_state.last_nav = step['text']
                    st.session_state.nav_idx += 1
        else:
            speak("Arrived")
            st.success("Destination reached")
            st.session_state.nav_active = False

    st.subheader("🎯 Objects Nearby")
    if ctx.video_processor:
        with ctx.video_processor.lock:
            objs = ctx.video_processor.detections.copy()
        if objs:
            st.write(", ".join([f"{v} {k}" for k,v in objs.items()]))
            now = time.time()
            for o in objs:
                if o not in st.session_state.detected_memory or now - st.session_state.detected_memory[o] > 4:
                    speak(o)
                    st.session_state.detected_memory[o] = now
