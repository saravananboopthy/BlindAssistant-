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
.mic-hint { font-size: 0.8rem; color: #64748b; margin-top: -10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# Persistent State
if "state" not in st.session_state:
    st.session_state.state = {
        "nav_steps": [], "nav_idx": 0, "active": False, 
        "last_nav_msg": "", "obj_memory": {}, "run_camera": True,
        "last_nav_time": 0, "active_nav_voice": "", "active_alert_voice": "", "voice_expiry": 0
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
        # Lowered confidence from 0.45 to 0.35 to catch MORE objects correctly on mobile cameras
        res = model(img, conf=0.35, verbose=False)[0]
        now = []
        h, w, _ = img.shape
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            label = model.names[int(b.cls[0])]
            pos = "left" if (x1+x2)/2 < w*0.33 else "right" if (x1+x2)/2 > w*0.66 else "ahead"
            width = x2 - x1
            if width > 400: dist = "very close"
            elif width > 200: dist = "near"
            else: dist = "far"
            now.append({"label": label, "pos": pos, "dist": dist})
        with self.lock: self.detections = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
            # Clean for speech: only take the part before any comma or city name
            speech_instr = instr.split(',')[0]
            steps.append({"text": instr, "speech": speech_instr, "lat": s["end_location"]["lat"], "lng": s["end_location"]["lng"]})
        return steps, None
    except Exception as e: return None, str(e)

def calculate_dist(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2); dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ==========================================
# UI
# ==========================================
st.markdown('<div class="main-header"><h1>👁️ Blind Assistant</h1></div>', unsafe_allow_html=True)

nav_instruction = ""
alert_instruction = ""

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

with col_c:
    st.subheader("Control Center")
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", os.getenv("GOOGLE_MAPS_API_KEY", ""))
    
    st.markdown('<p class="mic-hint">💡 Tap the input below and click your phone keyboard\'s 🎤 icon to speak your destination.</p>', unsafe_allow_html=True)
    dest_in = st.text_input("Destination (Type or use phone 🎤)", placeholder="e.g. Hope College", key="dest_field")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 ACTIVATE", use_container_width=True):
            if u_lat and api_key and dest_in:
                data, err = get_walking_directions((u_lat, u_lng), dest_in, api_key)
                if data:
                    st.session_state.state.update({"nav_steps": data, "nav_idx": 0, "active": True, "run_camera": True, "last_nav_msg": ""})
                    nav_instruction = "Navigation started. Path found."
                else: st.error(err)
    with c2:
        if st.button("🛑 STOP", type="primary", use_container_width=True):
            st.session_state.state.update({"active": False, "run_camera": False, "nav_steps": [], "nav_idx": 0})
            st.rerun()

    if st.session_state.state["active"] and u_lat:
        steps = st.session_state.state["nav_steps"]
        idx = st.session_state.state["nav_idx"]
        if idx < len(steps):
            cur_step = steps[idx]
            dist = calculate_dist(u_lat, u_lng, cur_step['lat'], cur_step['lng'])
            st.success(f"**Action:** {cur_step['text']}")
            
            now = time.time()
            
            # 1) Speak the navigation instruction if it hasn't been spoken yet
            if cur_step['text'] != st.session_state.state["last_nav_msg"]:
                nav_instruction = cur_step['speech']
                st.session_state.state["last_nav_msg"] = cur_step['text']
                st.session_state.state["last_nav_time"] = now
                
            # 2) Live GPS advance logic: Check if we arrived at the current step's physical location
            elif dist < 12 and (now - st.session_state.state["last_nav_time"]) > 15:
                if st.session_state.state["nav_idx"] < len(steps) - 1:
                    st.session_state.state["nav_idx"] += 1
                else:
                    nav_instruction = "Arrived at destination"
                    st.session_state.state["active"] = False

    st.divider()
    if st.session_state.state["run_camera"] and ctx.video_processor:
        with ctx.video_processor.lock:
            objs = ctx.video_processor.detections.copy()
        if objs:
            st.write(", ".join([f"{o['label']} at {o['pos']} ({o['dist']})" for o in objs]))
            now = time.time()
            for o in objs:
                tag = f"{o['label']}_{o['pos']}"
                # Universal strict 15-second cooldown
                if tag not in st.session_state.state["obj_memory"] or now - st.session_state.state["obj_memory"][tag] > 15:
                    label = o.get('label', 'object')
                    pos = o.get('pos', 'ahead')
                    dist = o.get('dist', 'near')
                    
                    if pos == "left":
                        alert_instruction = f"{label} is on your left, move right"
                    elif pos == "right":
                        alert_instruction = f"{label} is on your right, move left"
                    else: # ahead
                        alert_instruction = f"{label} is ahead, move left or right. It is {dist}."
                        
                    st.session_state.state["obj_memory"][tag] = now
                    break

# ==========================================
# MASTER SYNC VOICE QUEUE
# ==========================================
# Persist voice messages for a few seconds so they aren't cut off by faster reruns
if nav_instruction or alert_instruction:
    st.session_state.state["active_nav_voice"] = nav_instruction
    st.session_state.state["active_alert_voice"] = alert_instruction
    st.session_state.state["voice_expiry"] = time.time() + 8 # 8-second stability window

# Clear expired messages
if time.time() > st.session_state.state.get("voice_expiry", 0):
    st.session_state.state["active_nav_voice"] = ""
    st.session_state.state["active_alert_voice"] = ""

voice_hub = json.dumps({
    "nav": st.session_state.state.get("active_nav_voice", ""),
    "alert": st.session_state.state.get("active_alert_voice", "")
})

html_code = """
    <div style="background:#f1f5f9; border:1px solid #cbd5e1; padding:10px; border-radius:10px; text-align:center;">
        <button id="vbtn" onclick="lck()" style="background:#ef4444; color:white; border:none; padding:10px; border-radius:8px; cursor:pointer; font-weight:bold; width:100%; font-size:16px;">
            🔊 TAP TO SYNC BRAIN & VOICE 🚨
        </button>
    </div>
    <script>
    const vdata = VOICE_DATA_PLACEHOLDER;
    function lck() { localStorage.setItem('v_unlocked', 'true'); update(); speakQueue("Brain synchronized. Ready."); }
    function update() { if(localStorage.getItem('v_unlocked') === 'true') { let b = document.getElementById('vbtn'); b.style.background = "#10b981"; b.innerText = "✔️ VOICE SYNCED"; } }
    
    window.speechQueue = window.speechQueue || [];
    window.isSpeakingNow = window.isSpeakingNow || false;

    function processQueue() {
        if(window.isSpeakingNow || window.speechQueue.length === 0) return;
        window.isSpeakingNow = true;
        let t = window.speechQueue.shift();
        
        let u = new SpeechSynthesisUtterance(t);
        u.rate = 0.95; 
        
        // Find a Female Voice
        let voices = window.speechSynthesis.getVoices();
        let femaleVoice = voices.find(v => v.name.includes('Female') || v.name.includes('Zira') || v.name.includes('Samantha') || v.name.includes('Victoria') || v.name.includes('Google US English'));
        if (femaleVoice) {
            u.voice = femaleVoice;
        }
        
        u.onend = function() { window.isSpeakingNow = false; processQueue(); };
        u.onerror = function() { window.isSpeakingNow = false; processQueue(); };
        window.speechSynthesis.speak(u);
    }

    function speakQueue(t) {
        if(localStorage.getItem('v_unlocked') !== 'true') return;
        window.speechQueue.push(t);
        processQueue();
    }
    
    // Ensure voices are loaded (some browsers need this)
    window.speechSynthesis.onvoiceschanged = function() {
        // Optional: trigger a queue process if needed
    };

    update();
    
    if (vdata.nav || vdata.alert) {
        const h = vdata.nav + "|" + vdata.alert;
        if(window.lastH !== h) {
            // Add to Voice Queue sequentially. Navigation first, then Alerts.
            if (vdata.nav) {
                speakQueue(vdata.nav);
            }
            if (vdata.alert) {
                speakQueue(vdata.alert);
            }
            window.lastH = h;
        }
    }
    </script>
""".replace("VOICE_DATA_PLACEHOLDER", voice_hub)

components.html(html_code, height=85, key="voice_engine_stable")

if st.session_state.state["active"] or st.session_state.state["run_camera"]:
    time.sleep(1.2)
    st.rerun()
