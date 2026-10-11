# ♟️ MAGNUS — Gemini 3.8 Live Speech-to-Speech Voice Agent

**Magnus** is an autonomous roundtable character agent powered by Google's latest **Gemini 3.8 Live Speech-to-Speech (`gemini-3.8-live`)** model with continuous **Active Microphone Listening (VAD)** and a full **Web Control Center** where every option is controllable on screen in real time.

---

## ⚡ Quick Launch

From the root directory (`pass-the-mic/`) or inside `magnus_agent/`:

```powershell
# Launch the Magnus Web Control Center (opens at http://127.0.0.1:8050)
python magnus_agent/app.py --open-browser
```

Then open **`http://127.0.0.1:8050`** in your browser!

---

## 🖥️ Web Control Center Features (All Options on Screen)

1. **🤖 Gemini Speech-to-Speech Model & Voice Selector:**
   - **`gemini-3.8-live`** — Flagship real-time bidirectional Speech-to-Speech model (`bidiGenerateContent`)
   - **`gemini-3.8-live-extended-thinking`** — Real-time Speech-to-Speech with deep strategic reasoning
   - **`gemini-2.5-flash-native-audio-latest`** — Low-latency native audio stream over Gemini Live API
   - **`gemini-3.8-flash` + `gemini-3.8-flash-tts`** — Multimodal audio input + Gemini 3.8 Flash TTS
   - **Prebuilt Voice Profiles:** `Charon` (Deep & Commanding — Magnus Default), `Fenrir`, `Puck`, `Kore`, `Aoede`
   - **Automatic `.env` Key Loader + Live Hot-Swap:** Automatically reads `Gemini-API_KEY` / `GEMINI_API_KEY` from `.env`, and lets you update the key directly on screen.

2. **🎙️ Continuous Active Listening & Hardware Controls:**
   - **Animated Voice Orb & Live RMS Audio Meter:** Visualizes real-time microphone energy levels against the VAD threshold.
   - **3 Listening Modes:**
     - **🎯 Wake-Word Mode:** Continuously listens to and transcribes the room, waking up to speak when *"Magnus"* or *"Passing the mic to Magnus"* is heard.
     - **♾️ Always-On Mode:** 1-on-1 live Speech-to-Speech conversation — responds to every spoken utterance heard on the microphone.
     - **🖐️ Manual Mode:** Only speaks when triggered via button or prompt bar.
   - **Microphone Selector, VAD Threshold Slider, Silence Duration Slider, and 1-Click Room Noise Calibration.**

3. **♟️ Live Persona & Roundtable Configuration:**
   - Edit Magnus's name, debate topic (with quick preset dropdown), system instruction/persona, speaking style, wake phrases, and peer teams on the fly.
   - **Half-Duplex Echo Lock:** Automatically locks the microphone while Magnus is speaking through the laptop speakers so Magnus never hears his own voice.
