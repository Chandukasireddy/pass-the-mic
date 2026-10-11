# 🎙️ Attendee Workshop: Build Your Acoustic Voice Agent in 3 Minutes

Welcome to the **Pass-The-Mic** voice roundtable!

During the meetup, your laptop is an autonomous voice debater (e.g., **Team Orange**, **Team Red**, **Team Blue**) with **Ears (Microphone)** and a **Mouth (Speaker)**:
1. Your laptop **listens to the room through its physical Microphone** (using Voice Activity Detection + Speech-to-Text).
2. When the Moderator or another laptop says *"Passing the mic to Team Orange!"* out loud in the room, your laptop **hears its team name and wakes up**!
3. Your laptop thinks of a counter-argument to what it just heard on the mic, **speaks out loud through your laptop Speakers**, and ends with *"Passing the mic to Team Red!"* so the next laptop's microphone wakes up!

**Zero Wi-Fi setup, zero IP addresses, and zero firewall headaches — pure sound waves over the air!**

---

## ⚡ 3-Minute Quickstart

### Step 1: Install Requirements
Ensure your Python environment is active:
```bash
pip install -r ../requirements.txt
```

---

### Step 2: Test Your Speaker & Microphone Locally
Verify that your laptop speakers play clearly AND that your microphone hears your voice:
```bash
# 1. Test your laptop's Speaker output (Edge-TTS voice):
python test_local_voice.py --preset presets/orange_optimist.json

# 2. Test BOTH your Speaker AND your Microphone (records 4s of your voice & transcribes it!):
python test_local_voice.py --preset presets/orange_optimist.json --test-mic
```
To list all detected microphones/speakers or browse the voice catalog:
```bash
python test_local_voice.py --list-devices
python test_local_voice.py --list-voices
```

---

### Step 3: Craft Your Persona
Pick one of the presets in `presets/`, or create your own JSON file:
- **`presets/orange_optimist.json`** — Tech acceleration, open-source enthusiast, high energy (`en-US-AriaNeural`).
- **`presets/red_skeptic.json`** — Pragmatic critic, compute bottleneck auditor, sharp wit (`en-GB-RyanNeural`).
- **`presets/blue_philosopher.json`** — Contemplative thinker, ethics, human identity (`en-AU-NatashaNeural`).
- **`presets/emerald_builder.json`** — Hands-on engineer, production latency, shipping code (`en-US-ChristopherNeural`).

#### Creating Your Own Custom Character (`my_team.json`):
```json
{
  "name": "Team Emerald",
  "voice": "en-US-ChristopherNeural",
  "persona": "Hands-on Systems Engineer — focuses on real-world latency, production reliability, and shipping working code.",
  "style": "Direct, practical, fast-paced, pragmatic",
  "test_phrase": "Hey folks! Team Emerald here. My microphone and speaker are ready."
}
```

---

### Step 4: (Optional) Set Your Free Groq API Key
Pass-The-Mic supports [Groq's Free Tier](https://console.groq.com/keys) for both sub-second **Llama-3.3** responses and **Whisper** microphone transcription:
1. Copy `../.env.example` to `../.env` and set:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```
*(Note: Even without an API key, `core/stt.py` automatically uses **Google's 100% free, keyless SpeechRecognition API** for microphone transcription, and `core/llm.py` uses the built-in contextual simulation engine!)*

---

### Step 5: Run Your Agent at the Meetup Table!
Put your laptop on the table, turn your speaker volume up to 80–100%, unmute your microphone, and run:
```bash
python run_my_agent.py --preset presets/orange_optimist.json
```
Your terminal will calibrate to the room's ambient noise and start listening:
```text
🎙️ [Team Orange] Microphone listening to room...
   🌊 Voice detected in room! Recording speech until silence...
👂 Heard via Laptop Microphone: "Welcome to Pass-The-Mic! ... Passing the mic to Team Orange!"
🎤 WAKE TRIGGER DETECTED! THE MIC IS PASSED TO TEAM ORANGE!
🔊 Speaking Out Loud via Laptop Speaker...
✔ Finished speaking! Passed the mic over the air to Team Red.
```

---

## 🎭 Curated Voice Catalog

| Team / Alias | Voice ID | Accent / Gender | Tone |
| :--- | :--- | :--- | :--- |
| **Moderator** | `en-US-GuyNeural` | US English (Male) | Authoritative, broadcast-quality |
| **Team Orange** | `en-US-AriaNeural` | US English (Female) | Energetic, upbeat, dynamic |
| **Team Red** | `en-GB-RyanNeural` | British English (Male) | Analytical, crisp, sharp |
| **Team Blue** | `en-AU-NatashaNeural` | Australian (Female) | Contemplative, articulate |
| **Team Emerald** | `en-US-ChristopherNeural`| US English (Male) | Confident, direct, technical |
| **Team Purple** | `en-GB-SoniaNeural` | British English (Female)| Expressive, philosophical |
| **Team Amber** | `en-IN-NeerjaNeural` | Indian English (Female) | Warm, engaging, articulate |
| **Team Silver** | `en-US-JennyNeural` | US English (Female) | Natural, conversational |

---

## ❓ Troubleshooting at the Table

- **Laptop didn't wake up when called?**
  Turn up the speaking laptop's speaker volume so your microphone hears it clearly, or adjust silence sensitivity with `--silence-duration 1.5`.
- **Does my laptop hear its own voice?**
  No! `core/audio.py` enforces a strict **Half-Duplex Lock** — while your laptop's speaker is talking, its microphone is automatically muted plus a `0.35s` room-reverb cooldown.
