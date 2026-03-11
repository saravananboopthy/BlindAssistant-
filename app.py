"""
Blind Assistant - AI Vision + Navigation
Streamlit Cloud compatible
"""

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import threading
from collections import Counter
from ultralytics import YOLO
from navigator import get_walking_directions

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Blind Assistant",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------------------
# LOAD YOLO MODEL
# -----------------------------------------
@st.cache_resource
def load_model():
    # Auto-download from Ultralytics
    return YOLO("yolov8n")

model = load_model()

VEHICLE_OBJECTS = {"car","truck","bus","motorcycle","bicycle"}
WARNING_OBJECTS = {"dog","stop sign","traffic light"}

# -----------------------------------------
# RTC CONFIG
# -----------------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------------------
# SESSION STATE
# -----------------------------------------
for key, default in {
    "nav_steps": [],
    "nav_current": 0,
    "nav_active": False,
    "nav_summary": {},
    "last_spoken": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------------------
# VIDEO PROCESSOR
# -----------------------------------------
class BlindProcessor(VideoProcessorBase):

    confidence = 0.40

    def __init__(self):
        self.lock = threading.Lock()
        self.detections = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=self.confidence, verbose=False)[0]

        detected = []

        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            detected.append(label)

            if label in VEHICLE_OBJECTS:
                color = (0,0,255)
            elif label in WARNING_OBJECTS:
                color = (0,255,255)
            else:
                color = (0,255,0)

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

            text = f"{label} {conf:.0%}"
            cv2.putText(
                img,text,(x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,color,2
            )

        with self.lock:
            self.detections = dict(Counter(detected))

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------------------
# TEXT TO SPEECH
# -----------------------------------------
def browser_speak(text):

    escaped = text.replace('"', '\\"')

    components.html(
        f"""
<script>
var msg = new SpeechSynthesisUtterance("{escaped}");
speechSynthesis.speak(msg);
</script>
""",
        height=0
    )


# -----------------------------------------
# SIDEBAR
# -----------------------------------------
with st.sidebar:

    st.title("Blind Assistant")

    confidence = st.slider(
        "Detection Confidence",
        0.1,1.0,0.4
    )

    voice_enabled = st.toggle(
        "Voice Alerts"
    )

    st.divider()

    st.subheader("Navigation")

    source = st.text_input("Start location")
    destination = st.text_input("Destination")

    if st.button("Start Navigation"):

        if source and destination:

            result,error = get_walking_directions(
                source,
                destination
            )

            if result:
                st.session_state.nav_steps = result["steps"]
                st.session_state.nav_summary = result["summary"]
                st.session_state.nav_current = 0
                st.session_state.nav_active = True
            else:
                st.error(error)


# -----------------------------------------
# MAIN UI
# -----------------------------------------
st.title("👁️ Blind Assistant")

col1,col2 = st.columns([3,2])

# CAMERA
with col1:

    st.subheader("Camera")

    ctx = webrtc_streamer(
        key="blind-assistant",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=BlindProcessor,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True
    )

# DETECTIONS
with col2:

    st.subheader("Detected Objects")

    detection_box = st.empty()

    if ctx.state.playing and ctx.video_processor:

        with ctx.video_processor.lock:
            detections = ctx.video_processor.detections.copy()

        if detections:

            for obj,count in detections.items():

                st.write(f"⚠ {count} {obj}")

            if voice_enabled:

                text = ", ".join(
                    f"{count} {obj}" for obj,count in detections.items()
                )

                browser_speak(text)

        else:

            st.info("No objects detected")

    else:

        st.info("Start camera to begin detection")


# -----------------------------------------
# NAVIGATION UI
# -----------------------------------------
if st.session_state.nav_active:

    st.divider()

    st.subheader("Navigation")

    idx = st.session_state.nav_current
    steps = st.session_state.nav_steps

    if steps:

        step = steps[idx]

        st.success(step["text"])

        colA,colB = st.columns(2)

        if colA.button("Previous") and idx>0:
            st.session_state.nav_current -= 1
            st.rerun()

        if colB.button("Next") and idx < len(steps)-1:
            st.session_state.nav_current += 1
            st.rerun()
