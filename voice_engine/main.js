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

function updateStatus(msg, color='#666') {
    let el = document.getElementById("gps-tracker");
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
        // Prime the engine with a silent/brief utterance
        const p = new SpeechSynthesisUtterance("Assistant Active");
        p.volume = 0; // Silent prime
        window.speechSynthesis.speak(p);
        
        // Real welcome message
        setTimeout(() => speak("Voice Assistant Unlocked. Checking sensors.", true), 100);

        document.getElementById("unlock-btn").style.display = 'none';
        document.getElementById("gps-tracker").style.display = 'block';
        updateStatus("GPS: Seeking signal...");
        
        navigator.geolocation.watchPosition((p) => {
            window.currentLat = p.coords.latitude;
            window.currentLng = p.coords.longitude;
            updateStatus("GPS Live: " + p.coords.latitude.toFixed(4) + ", " + p.coords.longitude.toFixed(4), "#00d4aa");
        }, (e) => {
            updateStatus("GPS Error: " + e.message, "#ff6b6b");
        }, {enableHighAccuracy: true });
    } catch(e) {
        alert("Audio Init Error: " + e.message);
    }
}

window.unlockVoice = unlockVoice;

function speak(text, prio=false) {
    if (!window.voiceUnlocked) return;
    if (window.speechSynthesis.speaking && !prio) return;
    if (prio) window.speechSynthesis.cancel();
    let u = new SpeechSynthesisUtterance(text); 
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
}

// Navigation background check
setInterval(() => {
    if (window.currentSteps && window.currentSteps.length > 0 && window.navIdx < window.currentSteps.length && window.currentLat) {
        let t = window.currentSteps[window.navIdx];
        let d = calculateDistance(window.currentLat, window.currentLng, t.lat, t.lng);
        
        let now = Date.now();
        if ((now - window.lastNavSpeak) > 30000) {
           speak("Continue: " + t.text);
           window.lastNavSpeak = now;
        }

        if (d < 15) {
            window.navIdx++;
            if(window.navIdx < window.currentSteps.length) {
               speak("Now, " + window.currentSteps[window.navIdx].text, true);
               window.lastNavSpeak = Date.now();
            } else {
               speak("You have arrived.", true);
               window.currentSteps = [];
            }
        }
    }
}, 1000);

function onRender(event) {
    const data = event.data.args;
    let dets = data.detections || [];
    let steps = data.nav_steps || [];
    
    // Process new steps
    let shash = JSON.stringify(steps);
    if (shash !== window.currentRouteHash) {
        window.currentRouteHash = shash;
        window.currentSteps = steps;
        window.navIdx = 0;
        if(steps.length > 0) {
            speak("Navigation starting. " + steps[0].text, true);
            window.lastNavSpeak = Date.now();
        }
    }

    // Process Detections
    if (dets.length > 0 && data.engine_active) {
        let isDanger = false;
        let tParts = [];
        let seen = new Set();
        for (let d of dets) {
            if(!seen.has(d.label)) {
                tParts.push(d.label + " " + d.pos);
                seen.add(d.label);
                if(d.dist === 'near') isDanger = true;
            }
            if(tParts.length >= 2) break;
        }
        if(tParts.length > 0) {
            let msg = (isDanger ? "Watch out! " : "I see ") + tParts.join(" and ");
            let now = Date.now();
            // 9 second throttle
            if(window.lastSpeakStr !== msg || (now - window.lastSpeakTime > 9000)) {
                speak(msg, isDanger);
                window.lastSpeakStr = msg;
                window.lastSpeakTime = now;
            }
        }
    }

    sendMessageToStreamlit("setFrameHeight", {height: 120});
}

window.addEventListener("message", (e) => {
    if(e.data.type === "streamlit:render") {
        onRender(e);
    }
});

window.addEventListener("load", () => {
    sendMessageToStreamlit("setComponentReady", {apiVersion: 1});
});
