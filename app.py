import streamlit as st
import json
import time
import os
from PIL import Image

# Streamlit Configuration
st.set_page_config(
    page_title="Blind Assistant Monitor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium Look)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .nav-card {
        padding: 20px;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    .detection-card {
        background: #1f2937;
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 5px solid #10b981;
    }
    .detection-critical {
        border-left: 5px solid #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("👁️ Blind Assistant - Pro Monitoring")
st.write("---")

# Layout
col_feed, col_info = st.columns([3, 2])

# IPC File Paths
STATUS_FILE = "system_status.json"
FRAME_FILE = "latest_frame.jpg"

def load_system_data():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def load_latest_frame():
    if os.path.exists(FRAME_FILE):
        try:
            return Image.open(FRAME_FILE)
        except:
            return None
    return None

# Sidebar Controls
with st.sidebar:
    st.header("📊 System Stats")
    engine_status_placeholder = st.empty()
    st.write("---")
    st.info("💡 Start 'main.py' first to activate the engine.")

# Main Loop (Reactive UI)
# Note: Streamlit doesn't support built-in fast loops well without extra components,
# so we use a simple loop with st.empty placeholders.

with col_feed:
    st.subheader("📷 Live Computer Vision Feed")
    video_placeholder = st.empty()

with col_info:
    st.subheader("📍 Navigation & Logic")
    nav_placeholder = st.empty()
    st.write("---")
    st.subheader("🔍 Real-time Detections")
    detect_placeholder = st.empty()

# Persistent state for status
last_status = "Offline"

while True:
    data = load_system_data()
    img = load_latest_frame()
    
    if data:
        # Update Feed
        if img:
            video_placeholder.image(img, use_container_width=True)
        else:
            video_placeholder.warning("Camera feed signal low...")

        # Update Navigation Info
        with nav_placeholder.container():
            st.markdown(f"""
            <div class="nav-card">
                <h3>Current Status: {data.get('status', 'Unknown')}</h3>
                <p style="font-size: 1.2em;">Target: <b>{data.get('destination', 'No target set')}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            steps_total = data.get("total_steps", 0)
            if steps_total > 0:
                current_idx = data.get("nav_index", 0)
                progress = (current_idx + 1) / steps_total
                st.progress(min(progress, 1.0))
                st.write(f"**Next Action:** {data.get('current_step', 'N/A')}")
            else:
                st.write("Waiting for voice input destination...")

        # Update Detections
        with detect_placeholder.container():
            detections = data.get("detected", [])
            if detections:
                for obj in detections:
                    cls_name = "detection-critical" if obj['dist'] == "near" else "detection-card"
                    label_color = "#ef4444" if obj['dist'] == "near" else "#10b981"
                    st.markdown(f"""
                    <div class="{cls_name}">
                        <span style="color: {label_color}; font-weight: bold;">{obj['label'].upper()}</span> 
                        | Position: <b>{obj['pos']}</b> | Distance: <b>{obj['dist']}</b>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("Path is clear.")

    if data:
        # Destination Control in Sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🗺️ Manual Navigation")
        manual_dest = st.sidebar.text_input("Type Destination:", key="dest_txt")
        if st.sidebar.button("Send to Engine"):
            if manual_dest:
                with open("destination_input.txt", "w") as f:
                    f.write(manual_dest)
                st.sidebar.success("Sent!")
            else:
                st.sidebar.warning("Enter a place first.")

        video_placeholder.image(FRAME_FILE, use_column_width=True)
        engine_status_placeholder.success("Engine: ONLINE")
    else:
        video_placeholder.info("Waiting for engine to start. Run 'python main.py' to begin.")
        engine_status_placeholder.error("Engine: OFFLINE")

    # High frequency update for monitor
    time.sleep(0.1)