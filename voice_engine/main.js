function sendMessageToStreamlit(type, data) {
    if(window.parent) {
        window.parent.postMessage({isStreamlitMessage: true, type: type, ...data}, "*");
    }
}

window.voiceUnlocked = false;
window.lastSpeakStr = "";
window.lastSpeakTime = 0;
window.navIdx = 0;
window.currentRouteHash = "";
window.currentLat = null;
window.currentLng = null;
window.currentSteps = [];
window.lastNavSpeak = 0;

function updateStatus(msg, color='#64748b') {
    let el = document.getElementById("gps");
    if(el) {
        el.innerText = msg;
        el.style.color = color;
    }
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    let R = 6371e3; let p1 = lat1 * Math.PI/180; let p2 = lat2 * Math.PI/180;
    let dp = (lat2-lat1) * Math.PI/180; let dl = (lon2-lon1) * Math.PI/180;
    let a = Math.sin(dp/2) * Math.sin(dp/2) + Math.cos(p1) * Math.cos(p2) * Math.sin(dl/2) * Math.sin(dl/2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

function unlockVoice() {
    try {
        window.voiceUnlocked = true;
        const p = new SpeechSynthesisUtterance("Voice Assistant Active");
        p.rate = 1.0;
        window.speechSynthesis.speak(p);
        
        let btn = document.getElementById("unlock-btn");
        btn.classList.add("active");
        btn.innerText = "✔️ VOICE ASSISTANT LIVE";
        updateStatus("Sensors Linked. Awaiting Data.");
    } catch(e) {
        console.error(e);
    }
}

window.unlockVoice = unlockVoice;

function speak(text, prio=false) {
    if (!window.voiceUnlocked) return;
    // Don't interrupt if someone is already talking unless it's a priority alert
    if (window.speechSynthesis.speaking && !prio) return;
    if (prio) window.speechSynthesis.cancel();
    
    let u = new SpeechSynthesisUtterance(text); 
    u.rate = 1.0;
    window.speechSynthesis.speak(u);
}

// Navigation Loop (1s)
setInterval(() => {
    if (window.currentSteps && window.currentSteps.length > 0 && window.navIdx < window.currentSteps.length && window.currentLat) {
        let t = window.currentSteps[window.navIdx];
        let d = calculateDistance(window.currentLat, window.currentLng, t.lat, t.lng);
        
        let now = Date.now();
        // Periodic Nav Reminder (50s)
        if ((now - window.lastNavSpeak) > 50000) {
           speak("Stay on path: " + t.text);
           window.lastNavSpeak = now;
        }

        // Auto Advance (15m)
        if (d < 15) {
            window.navIdx++;
            if(window.navIdx < window.currentSteps.length) {
               speak("Now, " + window.currentSteps[window.navIdx].text, true);
               window.lastNavSpeak = Date.now();
            } else {
               speak("Destination reached. You have arrived.", true);
               window.currentSteps = [];
            }
        }
    }
}, 1000);

function onRender(event) {
    const data = event.data.args;
    let dets = data.detections || [];
    let steps = data.nav_steps || [];
    
    // 1. Sync GPS
    if (data.p_lat && data.p_lng) {
        window.currentLat = data.p_lat;
        window.currentLng = data.p_lng;
        updateStatus("Navigation Track Active", "#10b981");
    }

    // 2. Sync Route
    let shash = JSON.stringify(steps);
    if (shash !== window.currentRouteHash) {
        window.currentRouteHash = shash;
        window.currentSteps = steps;
        window.navIdx = 0;
        if(steps.length > 0) {
            speak("Route updated. " + steps[0].text, true);
            window.lastNavSpeak = Date.now();
        }
    }

    // 3. Sync Detections (Avoid Echo)
    if (dets.length > 0 && data.engine_active && window.voiceUnlocked) {
        let isDanger = false;
        let objects = [];
        let seen = new Set();
        
        for (let d of dets) {
            if(!seen.has(d.label)) {
                objects.push(d.label + " " + d.pos);
                seen.add(d.label);
                if(d.dist === 'near') isDanger = true;
            }
            if(objects.length >= 2) break; // Maximum 2 objects per sentence for clarity
        }
        
        if(objects.length > 0) {
            let combinedMsg = (isDanger ? "Watch out! " : "Ahead is ") + objects.join(" and ");
            let now = Date.now();
            
            // INCREASED THROTTLE: 15 seconds for unique messages
            if(window.lastSpeakStr !== combinedMsg || (now - window.lastSpeakTime > 15000)) {
                speak(combinedMsg, isDanger);
                window.lastSpeakStr = combinedMsg;
                window.lastSpeakTime = now;
            }
        }
    }

    sendMessageToStreamlit("setFrameHeight", {height: 130});
}

window.addEventListener("message", (e) => {
    if(e.data.type === "streamlit:render") onRender(e);
});

window.addEventListener("load", () => {
    sendMessageToStreamlit("setComponentReady", {apiVersion: 1});
});
