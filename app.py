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
        "last_nav_time": 0,
        "active_nav_voice": "", "nav_voice_token": "", "voice_nav_expiry": 0,
        "active_alert_voice": "", "alert_voice_token": "", "voice_alert_expiry": 0,
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
        self._frame_count = {}  # consecutive frame confirmation

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = model(img, conf=0.45, verbose=False)[0]
        h, w, _ = img.shape
        found_this_frame = set()
        candidates = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            if bw < 40 or bh < 40:
                continue
            label = model.names[int(b.cls[0])]
            cx = (x1 + x2) / 2
            pos = "left" if cx < w * 0.33 else "right" if cx > w * 0.66 else "ahead"
            if bw > 400:   dist = "very close"
            elif bw > 200: dist = "near"
            else:           dist = "far"
            key = f"{label}_{pos}"
            found_this_frame.add(key)
            self._frame_count[key] = self._frame_count.get(key, 0) + 1
            if self._frame_count[key] >= 2:
                candidates.append({"label": label, "pos": pos, "dist": dist})
        for key in list(self._frame_count.keys()):
            if key not in found_this_frame:
                self._frame_count[key] = 0
        with self.lock:
            self.detections = candidates
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# NAVIGATION LOGIC
# ==========================================
def calculate_dist(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def interpolate_waypoints(lat1, lon1, lat2, lon2, segment_m=10):
    total = calculate_dist(lat1, lon1, lat2, lon2)
    if total <= segment_m:
        return [{"lat": lat2, "lng": lon2, "dist_m": round(total)}]
    num = max(1, int(total / segment_m))
    points = []
    for i in range(1, num + 1):
        frac = i / num
        points.append({
            "lat": lat1 + frac * (lat2 - lat1),
            "lng": lon1 + frac * (lon2 - lon1),
            "dist_m": round(total / num)
        })
    return points

def get_walking_directions(source, dest, api_key):
    try:
        gmaps = googlemaps.Client(key=api_key)
        res = gmaps.directions(source, dest, mode="walking")
        if not res: return None, "No route found."
        leg = res[0]["legs"][0]
        micro_steps = []
        prev_lat, prev_lng = source if isinstance(source, tuple) else (source[0], source[1])
        for s in leg["steps"]:
            raw = re.sub(r"<.*?>", "", s["html_instructions"]).replace("&nbsp;", " ").strip()
            action = raw.split(',')[0]
            end_lat = s["end_location"]["lat"]
            end_lng = s["end_location"]["lng"]
            waypoints = interpolate_waypoints(prev_lat, prev_lng, end_lat, end_lng, segment_m=10)
            for i, wp in enumerate(waypoints):
                seg_m = wp["dist_m"]
                speech = f"{action}, {seg_m} meters" if i == 0 else f"Straight, {seg_m} meters"
                micro_steps.append({
                    "text": speech, "speech": speech,
                    "lat": wp["lat"], "lng": wp["lng"], "seg_m": seg_m
                })
            prev_lat, prev_lng = end_lat, end_lng
        return micro_steps, None
    except Exception as e:
        return None, str(e)

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
                else:
                    st.error(err)
    with c2:
        if st.button("🛑 STOP", type="primary", use_container_width=True):
            st.session_state.state.update({"active": False, "run_camera": False, "nav_steps": [], "nav_idx": 0})
            st.rerun()

    if st.session_state.state["active"] and u_lat:
        steps = st.session_state.state["nav_steps"]
        idx = st.session_state.state["nav_idx"]
        if idx < len(steps):
            cur_step = steps[idx]
            dist_to_wp = calculate_dist(u_lat, u_lng, cur_step['lat'], cur_step['lng'])
            dist_m = int(dist_to_wp)
            total_steps = len(steps)
            st.success(f"**Step {idx+1}/{total_steps}:** {cur_step['text']} | 📍 {dist_m}m away")

            now = time.time()
            time_since_last = now - st.session_state.state["last_nav_time"]

            if cur_step['text'] != st.session_state.state["last_nav_msg"]:
                nav_instruction = cur_step['speech']
                st.session_state.state["last_nav_msg"] = cur_step['text']
                st.session_state.state["last_nav_time"] = now
            elif dist_to_wp < 8:
                if idx < total_steps - 1:
                    st.session_state.state["nav_idx"] += 1
                    st.session_state.state["last_nav_msg"] = ""
                    st.session_state.state["last_nav_time"] = now
                else:
                    nav_instruction = "You have arrived at your destination."
                    st.session_state.state["active"] = False
            elif time_since_last > 10 and dist_m > 8:
                nav_instruction = f"{dist_m} meters remaining"
                st.session_state.state["last_nav_time"] = now

    st.divider()
    if st.session_state.state["run_camera"] and ctx.video_processor:
        with ctx.video_processor.lock:
            objs = ctx.video_processor.detections.copy()
        if objs:
            st.write(", ".join([f"{o['label']} at {o['pos']} ({o['dist']})" for o in objs]))
            now = time.time()
            for o in objs:
                tag = f"{o['label']}_{o['pos']}"
                if tag not in st.session_state.state["obj_memory"] or now - st.session_state.state["obj_memory"][tag] > 15:
                    label = o.get('label', 'object')
                    pos   = o.get('pos', 'ahead')
                    dist  = o.get('dist', 'near')
                    if pos == "left":
                        alert_instruction = f"{label} is on your left, move right"
                    elif pos == "right":
                        alert_instruction = f"{label} is on your right, move left"
                    else:
                        alert_instruction = f"{label} is ahead, move left or right. It is {dist}."
                    st.session_state.state["obj_memory"][tag] = now
                    break

# ==========================================
# MASTER SYNC VOICE ENGINE
# ==========================================
now_ts = int(time.time())

if nav_instruction:
    st.session_state.state["active_nav_voice"]  = nav_instruction
    st.session_state.state["nav_voice_token"]   = f"n{now_ts}"
    st.session_state.state["voice_nav_expiry"]  = now_ts + 10

if alert_instruction:
    st.session_state.state["active_alert_voice"] = alert_instruction
    st.session_state.state["alert_voice_token"]  = f"a{now_ts}"
    st.session_state.state["voice_alert_expiry"] = now_ts + 10

if now_ts > st.session_state.state.get("voice_nav_expiry", 0):
    st.session_state.state["active_nav_voice"] = ""
    st.session_state.state["nav_voice_token"]  = ""

if now_ts > st.session_state.state.get("voice_alert_expiry", 0):
    st.session_state.state["active_alert_voice"] = ""
    st.session_state.state["alert_voice_token"]  = ""

nav_txt   = st.session_state.state.get("active_nav_voice", "")
nav_tok   = st.session_state.state.get("nav_voice_token", "")
alert_txt = st.session_state.state.get("active_alert_voice", "")
alert_tok = st.session_state.state.get("alert_voice_token", "")
combo_key = f"{nav_tok}|{alert_tok}"

voice_html = ("""
<div style="background:#f1f5f9;border:1px solid #cbd5e1;padding:10px;
            border-radius:10px;text-align:center;">
  <button id="vbtn" onclick="doUnlock()"
    style="background:#ef4444;color:white;border:none;padding:11px;
           border-radius:8px;cursor:pointer;font-weight:bold;
           width:100%;font-size:16px;touch-action:manipulation;">
    🔊 TAP TO SYNC BRAIN &amp; VOICE 🚨
  </button>
</div>
<script>
(function() {
  var NAV_TXT = __NAV_TXT__;
  var ALT_TXT = __ALT_TXT__;
  var COMBO   = __COMBO__;

  function isUnlocked() { return localStorage.getItem('ba_unlocked') === '1'; }
  function isSeen(k)    { return localStorage.getItem('baSp_' + k) === '1'; }
  function markSeen(k)  {
    localStorage.setItem('baSp_' + k, '1');
    setTimeout(function() { localStorage.removeItem('baSp_' + k); }, 30000);
  }

  function getVoice() {
    var vs = window.speechSynthesis.getVoices();
    return vs.find(function(v) {
      return v.name.indexOf('Zira') >= 0 || v.name.indexOf('Samantha') >= 0 ||
             v.name.indexOf('Victoria') >= 0 || v.name.indexOf('Female') >= 0 ||
             v.name.indexOf('Google US English') >= 0;
    }) || vs[0] || null;
  }

  function speakOne(txt, then) {
    var u = new SpeechSynthesisUtterance(txt);
    u.rate = 0.92; u.volume = 1.0;
    var fv = getVoice(); if (fv) u.voice = fv;
    u.onend  = then || function(){};
    u.onerror= then || function(){};
    window.speechSynthesis.speak(u);
  }

  function playMessages() {
    if (!isUnlocked()) return;
    if (!COMBO || COMBO === '|' || isSeen(COMBO)) return;
    markSeen(COMBO);
    window.speechSynthesis.cancel();
    if (NAV_TXT) {
      speakOne(NAV_TXT, function() {
        if (ALT_TXT) { setTimeout(function(){ speakOne(ALT_TXT, null); }, 250); }
      });
    } else if (ALT_TXT) {
      speakOne(ALT_TXT, null);
    }
  }

  window.doUnlock = function() {
    localStorage.setItem('ba_unlocked', '1');
    var b = document.getElementById('vbtn');
    if (b) { b.style.background='#10b981'; b.innerText='✔️ VOICE SYNCED'; }
    window.speechSynthesis.cancel();
    speakOne('Voice ready. Navigation active.', function() { playMessages(); });
  };

  if (isUnlocked()) {
    var b = document.getElementById('vbtn');
    if (b) { b.style.background='#10b981'; b.innerText='✔️ VOICE SYNCED'; }
  }

  if (window.speechSynthesis.getVoices().length > 0) {
    playMessages();
  } else {
    window.speechSynthesis.onvoiceschanged = function() { playMessages(); };
    setTimeout(playMessages, 1500);
  }
})();
</script>
""".replace("__NAV_TXT__", json.dumps(nav_txt))
   .replace("__ALT_TXT__", json.dumps(alert_txt))
   .replace("__COMBO__",   json.dumps(combo_key)))

st.components.v1.html(voice_html, height=70)

if st.session_state.state["active"] or st.session_state.state["run_camera"]:
    time.sleep(1.2)
    st.rerun()
