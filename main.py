import cv2
import threading
import time
import queue
import os
import json
import asyncio
import requests
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import speech_recognition as sr
from dotenv import load_dotenv
import googlemaps

# Windows Geolocation fallback (optional if winsdk is installed)
try:
    from winsdk.windows.devices.geolocation import Geolocator
except ImportError:
    Geolocator = None

# Load environment variables
load_dotenv()

# ========================
# CONFIGURATION
# ========================
VOICE_PRIORITY = {
    "safety": 1,      # Objects extremely close
    "navigation": 2,  # Turn instructions
    "detection": 3    # General objects
}

# Detection Logic Constants
DIST_THRESHOLD_VERY_CLOSE = 350  # Pixels width (heuristic for distance)
DIST_THRESHOLD_NEAR = 200
DIST_THRESHOLD_MEDIUM = 100

# File paths for IPC
STATUS_FILE = "system_status.json"
FRAME_FILE = "latest_frame.jpg"

import math

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000  # Radius of earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ========================
# SYSTEM STATE
# ========================
class SystemState:
    def __init__(self):
        self.frame = None
        self.detected_objects = []
        self.current_location_coords = (0.0, 0.0)
        self.current_location = None
        self.destination = None
        self.nav_steps = []
        self.nav_index = 0
        self.status = "Initializing..."
        self.running = True
        self.vision_active = False # New flag
        self.lock = threading.Lock()
        self.last_save = 0

    def update_state(self, key, value):
        with self.lock:
            setattr(self, key, value)

    def save_to_disk(self):
        with self.lock:
            data = {
                "status": self.status,
                "destination": self.destination,
                "nav_index": self.nav_index,
                "total_steps": len(self.nav_steps),
                "current_step": self.nav_steps[self.nav_index]["instruction"] if self.nav_index < len(self.nav_steps) else "Arrived",
                "detected": self.detected_objects,
                "location": self.current_location
            }
            try:
                with open(STATUS_FILE, "w") as f:
                    json.dump(data, f)
                if self.frame is not None:
                    cv2.imwrite(FRAME_FILE, self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except: pass

state = SystemState()

# ========================
# VOICE SYSTEM
# ========================
class VoiceEngine:
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.last_spoken = {}  # {label: {pos, time}}
        self.debounce_counts = {} # {label_pos: count}
        self.last_global_speak = 0 # Prevent rapid-fire messages
        
    def worker(self):
        print("[VOICE] Fortifying Native Engine...")
        import pythoncom
        pythoncom.CoInitialize()
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
        except: return

        while state.running:
            try:
                priority, text = self.queue.get(timeout=1)
                
                # Tuning for Slower Speed as requested
                if priority == 1: speaker.Rate = 1   # Danger: Clear speed
                elif priority == 2: speaker.Rate = -1 # Nav: Slower for clarity
                else: speaker.Rate = -2               # Detection: Very slow and clear
                
                print(f"[VOICE] Speaking: {text}")
                speaker.Speak(text)
                self.queue.task_done()
            except queue.Empty: continue
            except: 
                pythoncom.CoInitialize() 
                continue

    def speak(self, text: str, priority: int = 3):
        # Prevent "next next" rapid fire - wait at least 1.5 seconds between different announcements
        now = time.time()
        if now - self.last_global_speak < 1.5 and priority > 1:
            return
        self.last_global_speak = now
        self.queue.put((priority, text))

    def should_speak_object(self, label, pos, dist):
        """Rules: Immediate for Danger, 4 frames for others."""
        now = time.time()
        key = f"{label}_{pos}"
        
        # ZERO DEBOUNCE FOR DANGER
        if dist == "very close":
            if key in self.last_spoken:
                if now - self.last_spoken[key] < 15: return False
            self.last_spoken[key] = now
            return True

        self.debounce_counts[key] = self.debounce_counts.get(key, 0) + 1
        # Increased to 5 frames to avoid 'fastly telling' every minor flicker
        if self.debounce_counts[key] < 5: 
            return False
            
        if key in self.last_spoken:
            if now - self.last_spoken[key] < 30:
                return False
        
        self.last_spoken[key] = now
        self.debounce_counts[key] = 0
        return True

voice = VoiceEngine()

# ========================
# NAVIGATION MODULE
# ========================
def get_live_location():
    """Triple-source high-speed location for India."""
    # Attempt high-accuracy Windows GPS/Wi-Fi location first
    if Geolocator is not None:
        try:
            async def get_windows_loc():
                locator = Geolocator()
                access = await Geolocator.request_access_async()
                if access == 1: # 1 = Allowed
                    pos = await locator.get_geoposition_async()
                    return (pos.coordinate.latitude, pos.coordinate.longitude)
                return None
            loc = asyncio.run(get_windows_loc())
            if loc: return loc
        except Exception:
            pass

    # Fallback to IP Geolocation
    with requests.Session() as s:
        # Source 1: ipwho.is (Excellent for Asia)
        try:
            r = s.get("http://ipwho.is/", timeout=1.5)
            if r.status_code == 200:
                d = r.json()
                return (d['latitude'], d['longitude'])
        except: pass

        # Source 2: ipapi.co
        try:
            r = s.get("https://ipapi.co/json/", timeout=1.5)
            if r.status_code == 200:
                d = r.json()
                return (d['latitude'], d['longitude'])
        except: pass
    
    return (13.0895, 80.2739) # Fallback

def get_route(source, destination):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key: return None, "API Key Missing"
    
    gmaps = googlemaps.Client(key=api_key)
    try:
        # SEARCH IN INDIA ONLY (region='in')
        places = gmaps.places(query=destination, location=(13.0, 80.0), radius=100000, region='in')
        if not places['results']:
            return None, f"Destination not found in India."
            
        best_match = places['results'][0]
        destination = best_match['formatted_address']
        print(f"[NAV] Target: {destination}")
        
        directions = gmaps.directions(source, destination, mode="walking", region='in')
        if not directions:
            return None, f"No walking route to {destination}"
        
        leg = directions[0]["legs"][0]
        steps = []
        for s in leg["steps"]:
            instr = s["html_instructions"].replace("<b>", "").replace("</b>", "").replace("<div style=\"font-size:0.9em\">", " ").replace("</div>", "")
            steps.append({
                "instruction": instr,
                "lat": s["end_location"]["lat"],
                "lng": s["end_location"]["lng"]
            })
        return steps, None
    except Exception as e:
        return None, f"Error: {str(e)}"

INPUT_FILE = "destination_input.txt"

def navigation_thread():
    state.status = "Fixing GPS..."
    last_coords = (0, 0)
    static_count = 0
    
    # Precise Start
    coords = get_live_location()
    state.update_state("current_location_coords", coords)
    state.update_state("current_location", f"{coords[0]}, {coords[1]}")

    voice.speak("System ready. You can speak your destination or type it in the dashboard.", priority=2)
    
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    
    dest = None
    while state.running and not dest:
        # Check for Typed Input from Dashboard
        if os.path.exists(INPUT_FILE):
             try:
                 with open(INPUT_FILE, "r") as f:
                     dest = f.read().strip()
                 os.remove(INPUT_FILE)
                 if dest: break
             except: pass

        # Voice Input
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                dest = recognizer.recognize_google(audio, language='en-IN').strip()
                if len(dest) < 3: dest = None
        except: continue

    voice.speak(f"Navigating to {dest}", priority=2)
    source_str = f"{coords[0]},{coords[1]}"
    steps, err = get_route(source_str, dest)
    
    if steps:
        # ACTIVATE VISION NOW
        state.update_state("vision_active", True)
        print("[NAV] Activating Vision System...")
        
        state.update_state("nav_steps", steps)
        state.update_state("status", "Navigating")
        
        for i, step in enumerate(steps):
            if not state.running: break
            state.update_state("nav_index", i)
            voice.speak(step["instruction"], priority=2)
            
            target_lat, target_lng = step["lat"], step["lng"]
            reached = False
            start_time = time.time()
            
            while not reached and state.running:
                curr = get_live_location()
                dist = haversine(curr[0], curr[1], target_lat, target_lng)
                
                # ETA & Navigation Details
                print(f"[NAV] {dist:.1f}m to turn. Next: {step['instruction'][:20]}...")
                
                if dist < 12: reached = True
                if time.time() - start_time > 45: reached = True 
                time.sleep(4)
            
            if reached: voice.speak("Waypoint reached.", priority=2)
        
        voice.speak("Navigation complete.", priority=2)
        state.update_state("status", "Arrived")
    else:
        voice.speak(f"Check destination. {err}", priority=1)
        threading.Thread(target=navigation_thread, daemon=True).start()

# ========================
# VISION MODULE
# ========================
def vision_thread():
    model = YOLO("yolov8s.pt") 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent camera frame buffer lag
    
    # Sequential Startup: Wait for navigation to begin
    print("[VISION] Waiting for destination to activate...")
    while state.running and not state.vision_active:
        cap.read() # Burn frames to keep real-time
        time.sleep(0.5)

    print("[VISION] Sensor Active. Scanning environment...")
    
    # Memory for "Behind" logic
    objects_in_front = {} # {label: time_last_seen_very_close}
    
    last_save_time = 0
    frame_count = 0
    while state.running:
        ret, frame = cap.read()
        if not ret: continue
        
        # Skip for CPU efficiency
        frame_count += 1
        if frame_count % 3 != 0: continue
        
        results = model(frame, conf=0.30, verbose=False)[0]
        h, w, _ = frame.shape
        detected_this_frame = []
        current_frame_labels = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            # Filter parking meter/traffic light noise in home settings
            if box.conf[0] < 0.45 and label in ["parking meter", "traffic light", "fire hydrant"]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) / 2
            
            # Descriptive Spatial Phrasing
            if x_center < w * 0.33: pos = "on your Left"
            elif x_center > w * 0.66: pos = "on your Right"
            else: pos = "Immediately Ahead"
            
            box_w = x2 - x1
            if box_w > DIST_THRESHOLD_VERY_CLOSE:
                 dist = "very close"
                 objects_in_front[label] = time.time() # Remember it 
            elif box_w > DIST_THRESHOLD_NEAR: dist = "near"
            else: dist = "far"
            
            detected_this_frame.append({"label": label, "pos": pos, "dist": dist})
            current_frame_labels.add(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # "Behind" / "Passed" Logic
        now = time.time()
        for label in list(objects_in_front.keys()):
            # If it was very close < 3s ago but is NOT seen now, it just passed behind
            if label not in current_frame_labels and (now - objects_in_front[label] < 2.5):
                voice.speak(f"{label} passed. Behind you.", priority=3)
                del objects_in_front[label]
            elif now - objects_in_front[label] > 4: # Forgot old ones
                del objects_in_front[label]

        for obj in detected_this_frame:
            if voice.should_speak_object(obj['label'], obj['pos'], obj['dist']):
                priority = 1 if obj['dist'] == 'very close' else 3
                msg = f"{obj['label']} {obj['pos']}"
                if priority == 1: msg = f"Watch out! {obj['label']} {obj['pos']}"
                voice.speak(msg, priority=priority)

        state.update_state("frame", frame)
        state.update_state("detected_objects", detected_this_frame)
        if time.time() - last_save_time > 0.1:
            state.save_to_disk()
            last_save_time = time.time()
        time.sleep(0.01)

    cap.release()

# ========================
# MAIN ENTRY
# ========================
def main():
    print("--- Blind Assistant Real-Time Engine ---")
    
    # Initialize files
    if os.path.exists(STATUS_FILE): os.remove(STATUS_FILE)
    if os.path.exists(FRAME_FILE): os.remove(FRAME_FILE)

    # Start independent modules
    threads = [
        threading.Thread(target=voice.worker, name="VoiceThread", daemon=True),
        threading.Thread(target=vision_thread, name="VisionThread", daemon=True),
        threading.Thread(target=navigation_thread, name="NavThread", daemon=True)
    ]
    
    for t in threads:
        t.start()
        print(f"Started {t.name}")
    
    try:
        while state.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system...")
        state.running = False
        time.sleep(1)

if __name__ == "__main__":
    main()