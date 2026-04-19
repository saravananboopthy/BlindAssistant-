# 👁️ Blind Assistant - Cloud Edition

A production-grade, real-time AI assistant designed for the visually impaired. This application leverages YOLOv8 for object detection and Google Maps for high-precision walking navigation, all accessible via a smooth, responsive Streamlit web interface.

## 🚀 Key Features

- **Real-time Object Detection**: Uses YOLOv8 to identify obstacles and hazards.
- **Dynamic Positioning**: Tells the user if an object is on their left, right, or straight ahead.
- **Smart Navigation**: Integrated Google Maps walking directions with step-by-step voice guidance.
- **Browser-Native TTS**: Utilizes the browser's speech synthesis for low-latency, adjustable voice feedback.
- **Glassmorphism UI**: A modern, premium interface designed for accessibility and clarity.
- **High-Accuracy Geolocation**: Fetches real-time location directly from the browser for precise navigation.

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS and JavaScript components.
- **Computer Vision**: Ultralytics YOLOv8 (yolov8n).
- **Navigation**: Google Maps Directions API.
- **WebRTC**: `streamlit-webrtc` for high-performance camera streaming.
- **Deployment**: Streamlit Cloud ready.

## 📋 Prerequisites

- Python 3.9+
- [Google Maps API Key](https://console.cloud.google.com/google/maps-apis/credentials) with Directions API enabled.

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saravananboopthy/BlindAssistant-.git
   cd BlindAssistant-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_MAPS_API_KEY=your_api_key_here
   ```

## 🏃 Running the App

```bash
streamlit run app.py
```

## 🏗️ Deployment (Streamlit Cloud)

1. Push your code to GitHub.
2. Connect your repo to Streamlit Cloud.
3. In the "Advanced settings", add your `GOOGLE_MAPS_API_KEY` to the `Secrets` section.
4. Deployment will automatically handle the `packages.txt` for system-level dependencies (ffmpeg, libgl1).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
