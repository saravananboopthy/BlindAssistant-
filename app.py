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
import json
from streamlit_geolocation import streamlit_geolocation

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Blind Assistant", page_icon="👁️", layout="wide")

# Custom CSS
st.markdown("""
<style>
.det-card { padding: 10px; border-radius: 8px; margin-bottom: 5px; background: rgba(0,212,170,0.1); border-left: 5px solid #00d4aa; }
.danger { background: rgba(255,107,107,0.1); border-left-color: #ff6b6b; color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "nav_steps" not in st.session_state:
    st.session_state.update({"nav_steps": [], "nav_idx": 0, "dest": "", "engine_active": False})

@st.cache_resource
def load_yolo():
    # Upgrade to 's' model for better accuracy
    return YOLO("yolov8s.pt")

model = load_yolo()

class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model:
            res = model(img, conf=0.5, verbose=False)[0] 
            now = []
            h, w, _ = img.shape
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                label = model.names[int(b.cls[0])]
                pos = "left" if (x1+x2)/2 < w*0.33 else "right" if (x1+x2)/2 > w*0.66 else "ahead"
                dist = "near" if (x2-x1) > 280 else "far"
                now.append({"label": label, "pos": pos, "dist": dist})
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,212,170), 2)
            with self.lock: self.detections = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("👁️ Blind Assistant")

# Real Geolocation fetching
st.write("### 📍 Location")
loc_col, stat_col = st.columns([1, 2])
with loc_col:
    location = streamlit_geolocation()

user_lat = location.get('latitude') if location else None
user_lng = location.get('longitude') if location else None

with stat_col:
    if user_lat:
        st.success(f"**GPS Locked:** {user_lat:.4f}, {user_lng:.4f}")
    else:
        st.error("**GPS Status:** Click the crosshair icon to lock your Starting Location")

# ==============================================================================
# JAVASCRIPT MASTER ENGINE (Handles Auto-Navigation and Throttled Voice without Page Reloads)
# ==============================================================================
MASTER_ENGINE = """
<script>
if (!window.masterEngineRunning) {
    window.masterEngineRunning = true;
    window.lastSpeakStr = "";
    window.lastSpeakTime = 0;
    window.currentLat = null;
    window.currentLng = null;
    window.lastNavSpeak = 0;
    window.navStarted = false;
    window.currentRouteHash = "";
    window.audioUnlocked = false; // Prevents browser blocking
    
    // Auto-Tracking GPS for Navigation
    navigator.geolocation.watchPosition(
        (p) => { 
            window.currentLat = p.coords.latitude; 
            window.currentLng = p.coords.longitude; 
            if (document.getElementById('gps-tracker')) document.getElementById('gps-tracker').innerText = "Live GPS Tracking: Active";
        },
        (e) => { 
            if (document.getElementById('gps-tracker')) document.getElementById('gps-tracker').innerText = "GPS Error: " + e.message; 
        },
        { enableHighAccuracy: true, maximumAge: 0 }
    );

    window.unlockVoice = function() {
        window.audioUnlocked = true;
        let u = new SpeechSynthesisUtterance("Voice Assistant Enabled.");
        u.rate = 0.9;
        window.speechSynthesis.speak(u);
        document.getElementById('unlock-btn').style.display = 'none';
        document.getElementById('gps-tracker').style.display = 'block';
    }

    function speak(text, prio=false) {
        if (!window.audioUnlocked) return; // Browser will block if not explicitly unlocked
        if (window.speechSynthesis.speaking && !prio) return;
        if (prio) window.speechSynthesis.cancel();
        let u = new SpeechSynthesisUtterance(text);
        u.rate = 0.9;
        window.speechSynthesis.speak(u);
    }

    function calculateDistance(lat1, lon1, lat2, lon2) {
        let R = 6371e3;
        let p1 = lat1 * Math.PI/180, p2 = lat2 * Math.PI/180;
        let dp = (lat2-lat1) * Math.PI/180, dl = (lon2-lon1) * Math.PI/180;
        let a = Math.sin(dp/2) * Math.sin(dp/2) + Math.cos(p1) * Math.cos(p2) * Math.sin(dl/2) * Math.sin(dl/2);
        return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    }

    setInterval(() => {
        let rawSteps = localStorage.getItem('nav_steps') || "[]";
        let steps = JSON.parse(rawSteps);
        let idx = parseInt(localStorage.getItem('nav_idx') || "0");
        let engineActive = localStorage.getItem('engine_active') === 'true';

        // 1. Navigation Logic (Auto Progress)
        if (rawSteps !== window.currentRouteHash && steps.length > 0) {
            window.currentRouteHash = rawSteps;
            window.navStarted = true;
            window.lastNavSpeak = Date.now();
            speak("Route found. Start moving. " + steps[0].text, true);
        }

        if (steps.length > 0 && window.currentLat && idx < steps.length) {
            let t = steps[idx];
            let dist = calculateDistance(window.currentLat, window.currentLng, t.lat, t.lng);
            
            // Remind every 30 seconds
            if (Date.now() - window.lastNavSpeak > 30000) {
                speak("Continue: " + t.text);
                window.lastNavSpeak = Date.now();
            }

            // Move to Next Step
            if (dist < 15) {
                idx++;
                localStorage.setItem('nav_idx', idx);
                if (idx < steps.length) {
                    speak("Now, " + steps[idx].text, true);
                    window.lastNavSpeak = Date.now();
                } else {
                    speak("You have arrived at your destination.", true);
                    localStorage.setItem('nav_steps', "[]");
                }
            }
        }

        // 2. Vision Logic (Combine multiple objects, prevent echo)
        if (engineActive) {
            let dets = JSON.parse(localStorage.getItem('vision_dets') || "[]");
            if (dets.length > 0) {
                let isDanger = false;
                let textParts = [];
                let seen = new Set();
                
                for (let d of dets) {
                    // Combine multiple objects
                    if (!seen.has(d.label)) {
                        textParts.push(d.label + " " + d.pos);
                        seen.add(d.label);
                        if (d.dist === 'near') isDanger = true;
                    }
                    if (textParts.length >= 2) break; // Limit to 2 objects max at once
                }
                
                if (textParts.length > 0) {
                    let msgText = (isDanger ? "Watch out, " : "I see ") + textParts.join(" and ");
                    const now = Date.now();
                    
                    // Throttle identical completely for 9 seconds to prevent Echo
                    if (window.lastSpeakStr !== msgText || (now - window.lastSpeakTime > 9000)) {
                        speak(msgText, isDanger);
                        window.lastSpeakStr = msgText;
                        window.lastSpeakTime = now;
                    }
                }
            }
        }
    }, 1000);
}
</script>
<div style="background:#f0f2f6; border-radius:10px; padding:15px; text-align:center;">
    <b>🧠 Background Voice Engine</b><br>
    <button id="unlock-btn" onclick="window.unlockVoice()" style="background:#ff6b6b; color:white; border:none; padding:10px 20px; border-radius:8px; cursor:pointer; font-weight:bold; margin-top:10px; font-size:16px; width:100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        🚨 TAP HERE TO UNLOCK VOICE 🔊
    </button>
    <span id="gps-tracker" style="display:none; font-size:12px; color:#666; margin-top:5px;">GPS Tracking initializing...</span>
</div>
"""
# Render the static engine exactly once. Height increased to fit the button.
components.html(MASTER_ENGINE, height=120)
# ==============================================================================

col_v, col_i = st.columns([1.5, 1])
with col_v:
    ctx = webrtc_streamer(key="v", video_processor_factory=VisionProcessor, 
                         rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
    
    if ctx.video_processor:
        st.session_state.engine_active = st.toggle("Activate Voice engine", value=st.session_state.engine_active)
        
        # Inject fast-changing data blindly into LocalStorage for the Master Engine to pick up
        current_dets = []
        with ctx.video_processor.lock: 
            current_dets = ctx.video_processor.detections.copy()
            
        inj = f"""<script>
        localStorage.setItem('vision_dets', '{json.dumps(current_dets)}');
        localStorage.setItem('engine_active', '{str(st.session_state.engine_active).lower()}');
        </script>"""
        components.html(inj, height=0)

with col_i:
    st.subheader("🧭 Path Input")
    
    # Safely load the API Key
    default_api = os.getenv("GOOGLE_MAPS_API_KEY", "")
    try:
        if st.secrets.get("GOOGLE_MAPS_API_KEY"):
            default_api = st.secrets["GOOGLE_MAPS_API_KEY"]
    except: pass
    
    api = st.text_input("G-Maps Key", value=default_api, type="password")
    
    # User uses built-in OS Voice dictation on this text field
    st.info("💡 Tap the text box below and use your phone's built-in 🎤 microphone button to speak your destination.")
    dest = st.text_input("Destination", placeholder="e.g. Hospital")
    
    if st.button("Start Navigation", type="primary"):
        if user_lat and api and dest:
            try:
                g = googlemaps.Client(key=api)
                r = g.directions((float(user_lat), float(user_lng)), dest, mode="walking")
                if r:
                    steps = []
                    for s in r[0]["legs"][0]["steps"]:
                        text = s["html_instructions"].replace("<b>","").replace("</b>","")
                        lat = s["end_location"]["lat"]
                        lng = s["end_location"]["lng"]
                        steps.append({"text": text, "lat": lat, "lng": lng})
                    
                    st.success("Route Generated! Voice Navigation Active.")
                    # Inject steps into localstorage for engine
                    inj_steps = f"""<script>
                    localStorage.setItem('nav_steps', '{json.dumps(steps)}');
                    localStorage.setItem('nav_idx', '0');
                    </script>"""
                    components.html(inj_steps, height=0)
                else: 
                    st.error("No path found.")
            except Exception as e: 
                st.error(f"Error: {e}")

    st.subheader("🎯 Active Detections")
    if 'current_dets' in locals() and current_dets:
        for d in current_dets:
            st.markdown(f'<div class="det-card {"danger" if d["dist"]=="near" else ""} "><b>{d["label"].upper()}</b> {d["pos"]}</div>', unsafe_allow_html=True)
            
if st.session_state.engine_active:
    time.sleep(1)
    st.rerun()
