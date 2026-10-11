# 🎙️ pass-the-mic: Multi-Agent Voice Roundtable (Speaker 🔊 + Microphone 🎙️ Edition)

A clean, production-ready Python repository for running live, in-person multi-agent voice roundtable debates where **laptops talk out loud through their physical Speakers and listen to each other through their physical Microphones** — no Wi-Fi coordination required!

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                   PASS-THE-MIC ACOUSTIC TOPOLOGY                           │
│           (Sound Waves in the Room = The Communication Network!)           │
│                                                                            │
│   ┌──────────────────────┐                     ┌───────────────────────┐   │
│   │    HOST MODERATOR    │   🔊 Sound Waves    │      TEAM ORANGE      │   │
│   │    ("Team Black")    │ ──────────────────► │    (Tech Optimist)    │   │
│   │  Speaker 🔊 + Mic 🎙️  │ "Passing the mic    │  Mic 🎙️ hears wake word│   │
│   └──────────▲───────────┘   to Team Orange!"  └───────────┬───────────┘   │
│              │                                             │               │
│              │                                             │ 🔊 Speaks out │
│              │ 3. Moderator Mic 🎙️                         │    loud & says│
│              │    monitors room &                          │  "Passing the │
│              │    wraps up debate                          │   mic to Red!"│
│              │                                             ▼               │
│              │                                 ┌───────────────────────┐   │
│              └──────────────────────────────── │       TEAM RED        │   │
│                     🔊 Sound Waves             │  (Pragmatic Skeptic)  │   │
│               "Passing the mic to Team Blue!"  │  Mic 🎙️ wakes & speaks │   │
│                                                └───────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 1. Project Vision & Meetup Dynamics

**Pass-The-Mic** turns a table of laptops into an autonomous, over-the-air voice roundtable where laptops literally converse out loud across the room:

1. **The Host (Repository Owner)** runs the **Moderator Agent** ("Team Black") via [`moderator/run_moderator.py`](file:///k:/Tech/Projects/pass-the-mic/moderator/run_moderator.py). The host's laptop speaks the debate topic out loud through its **Speaker** in an authoritative broadcast voice, calls on the first team (*"Passing the mic to Team Orange!"*), and then switches its **Microphone** on to listen to the room, keep time, and intervene out loud if a handoff is missed.
2. **Meetup Attendees** clone this repository and go straight to [`character_agents/`](file:///k:/Tech/Projects/pass-the-mic/character_agents). Each attendee picks a persona (e.g., **Team Orange**, **Team Red**, **Team Blue**), chooses a neural voice, tests their laptop speaker & microphone locally, and runs [`run_my_agent.py`](file:///k:/Tech/Projects/pass-the-mic/character_agents/run_my_agent.py).
3. **The Acoustic Turn-Taking Mechanic (Speaker 🔊 ➔ Air 🌊 ➔ Microphone 🎙️):**
   - Every attendee laptop continuously listens to the room through its **Microphone** using Voice Activity Detection (VAD) and Speech-to-Text (`core/stt.py`).
   - **Moderator's Speaker** announces: *"Welcome to Pass-The-Mic! ... Passing the mic to Team Orange!"*
   - **Team Orange's Microphone** hears *"Passing the mic to Team Orange"*, wakes up, feeds what it heard into its LLM, and **speaks out loud through its Speaker**, ending with: *"I believe open-source will win, and I'm passing the mic to Team Red!"*
   - **Team Red's Microphone** hears Team Orange's argument and the wake phrase *"passing the mic to Team Red"*, wakes up, and speaks its counter-argument out loud through its own **Speaker**!
   - **Half-Duplex Self-Mute:** While a laptop's speaker is talking, its own microphone is automatically locked (with a `0.35s` reverb cooldown) so it never hears its own voice.

---

## 🏗️ 2. Repository Architecture

```
pass-the-mic/
├── README.md                      # Overview of the acoustic Speaker + Mic architecture & meetup guide
├── requirements.txt              # Lightweight dependencies (edge-tts, pygame, sounddevice, SpeechRecognition, groq)
├── .env.example                  # GROQ_API_KEY / Mic VAD & STT settings
├── simulate_roundtable.py        # [HOST TEST] Simulates 1 Moderator + 2-3 Characters on ONE laptop
│
├── core/                         # Shared audio, speech, and LLM utilities
│   ├── __init__.py
│   ├── llm.py                    # Fast LLM client (Groq Llama-3 free tier, OpenAI & Mock fallback)
│   ├── tts.py                    # Mouth: Voice generator (edge-tts — 100% free, zero API keys needed)
│   ├── stt.py                    # Ears: Speech-to-Text (Groq Whisper + 100% free Google Web Speech API)
│   ├── audio.py                  # Speaker playback (pygame) + Microphone VAD recording (sounddevice)
│   └── protocol.py               # Acoustic wake-word/homophone detector + JSON schema for mic-passing
│
├── moderator/                    # [HOST CORE] Main engine run by the meetup host
│   ├── __init__.py
│   ├── run_moderator.py          # Acoustic Host runner (Speaks intro on Speaker, listens to room on Mic)
│   ├── moderator_agent.py        # The Host's Voice Agent (introduces topic, listens, intervenes, summarizes)
│   ├── dashboard.py              # Pretty terminal UI (rich) showing active speaker and live transcripts
│   ├── server.py                 # Optional hybrid WebSocket coordinator for noisy environments
│   └── topics.json               # Curated list of fun tech/AI debate topics
│
└── character_agents/             # [ATTENDEE WORKSHOP FOLDER] Clean template for participants
    ├── README.md                 # "How to build your Acoustic Voice Agent in 3 minutes"
    ├── run_my_agent.py           # Listens via laptop Microphone (STT), wakes on team name, speaks via Speaker!
    ├── test_local_voice.py       # Standalone script: tests both Speaker output (TTS) & Microphone input (STT)
    ├── agent_builder.py          # Modular participant agent class with listen_on_mic() and speak_turn()
    └── presets/                  # Pre-configured example characters for attendees to inspect
        ├── orange_optimist.json  # Tech Optimist persona & voice config
        ├── red_skeptic.json      # Pragmatic Skeptic persona & voice config
        ├── blue_philosopher.json # Deep Thinker persona & voice config
        └── emerald_builder.json  # Pragmatic Builder persona & voice config
```

---

## ⚡ 3. Core Technical Specifications

### A. Zero-Cost Speaker Output (`core/tts.py` & `core/audio.py`)
- Powered by `edge-tts` (Microsoft Edge Neural Voices). Requires **no API keys, no subscriptions, has zero rate limits**, and sounds remarkably human.
- Curated voice dictionary:
  - **Moderator:** `en-US-GuyNeural` (Authoritative, clear broadcast tone)
  - **Team Orange:** `en-US-AriaNeural` (Energetic, upbeat)
  - **Team Red:** `en-GB-RyanNeural` (British, analytical, sharp)
  - **Team Blue:** `en-AU-NatashaNeural` (Australian, relaxed, articulate)
  - **Team Emerald:** `en-US-ChristopherNeural` (Confident, direct)

### B. Zero-Cost Microphone Listening & Speech-to-Text (`core/audio.py` & `core/stt.py`)
- **Voice Activity Detection (VAD):** `core/audio.py` uses `sounddevice` to continuously monitor microphone RMS energy, automatically starting recording when someone in the room speaks and stopping after `1.8s` of silence.
- **Multi-Backend Speech-to-Text (`core/stt.py`):**
  1. **Groq Whisper (`whisper-large-v3-turbo`):** Sub-second transcription when `GROQ_API_KEY` is configured.
  2. **Google Web Speech API (via `SpeechRecognition`):** **100% free and requires zero API keys**, working out-of-the-box on any laptop!
  3. **OpenAI Whisper (`whisper-1`):** Optional fallback.

### C. Acoustic Wake-Phrase Detection & Turn Protocol (`core/protocol.py`)
- Every agent ends its spoken turn by clearly saying *"Passing the mic to [Next Team]!"* out loud through its speaker.
- `detect_mic_pass_in_transcript()` in [`core/protocol.py`](file:///k:/Tech/Projects/pass-the-mic/core/protocol.py) scans microphone STT transcripts with built-in **homophone tolerance** (e.g., matching `"team read"` ➔ `"Team Red"`, `"team blew"` ➔ `"Team Blue"`, `"passing the mike"` ➔ `"passing the mic"`).
- Internally, LLM responses follow a strict JSON schema:
  ```json
  {
    "speaker": "Team Orange",
    "spoken_text": "I believe open-source agents are moving faster than closed labs. Passing the mic to Team Red!",
    "next_agent": "Team Red"
  }
  ```

---

## 🚀 4. Quickstart Guide

### Option A: Simulate on 1 Laptop (`simulate_roundtable.py`)
Test the entire multi-agent voice roundtable on a single machine:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation (Speaks out loud through your speakers in 3 distinct voices!)
python simulate_roundtable.py --rounds 3

# Speak into your OWN laptop microphone in Round 1 and pass the mic to an AI agent!
python simulate_roundtable.py --rounds 3 --live-mic

# Fast muted text-only test:
python simulate_roundtable.py --mute --rounds 2
```

---

### Option B: For Meetup Attendees (Running Your Laptop Agent)

1. Navigate to [`character_agents/`](file:///k:/Tech/Projects/pass-the-mic/character_agents):
   ```bash
   cd character_agents
   ```
2. **Test your Speaker AND your Microphone locally:**
   ```bash
   # Test Speaker output (TTS)
   python test_local_voice.py --preset presets/orange_optimist.json

   # Test both Speaker AND Microphone (records 4s from your mic and reads it back!)
   python test_local_voice.py --preset presets/orange_optimist.json --test-mic
   ```
3. **Start listening and debating at the table:**
   ```bash
   python run_my_agent.py --preset presets/orange_optimist.json
   ```
   Your laptop will calibrate to ambient room noise, listen via its **Microphone**, wake up whenever someone in the room says *"Passing the mic to Team Orange!"*, and respond out loud via its **Speaker**!

---

### Option C: For the Meetup Host (Running the Moderator Laptop)

Place the Host laptop at the head of the table with speaker volume at 80–100% and run:
```bash
python moderator/run_moderator.py --teams "Team Orange,Team Red,Team Blue" --rounds 6
```
The Host laptop will:
1. **Speak** the debate topic out loud and pass the mic to the opening team.
2. **Listen** via its microphone to transcribe each laptop's speech in the room.
3. **Intervene out loud** if the room goes silent or a laptop misses a wake phrase.
4. **Conclude and summarize** the debate out loud after all rounds finish!

---

## 🎭 5. Curated Voice Catalog

| Team / Alias | Voice ID | Accent & Gender | Tone Description |
| :--- | :--- | :--- | :--- |
| **Moderator** | `en-US-GuyNeural` | US English (Male) | Authoritative, clear broadcast tone |
| **Team Orange** | `en-US-AriaNeural` | US English (Female) | Energetic, upbeat, dynamic |
| **Team Red** | `en-GB-RyanNeural` | British English (Male) | Analytical, crisp, sharp, skeptical |
| **Team Blue** | `en-AU-NatashaNeural` | Australian English (Female) | Contemplative, relaxed, articulate |
| **Team Emerald** | `en-US-ChristopherNeural` | US English (Male) | Confident, direct, technical |
| **Team Purple** | `en-GB-SoniaNeural` | British English (Female) | Expressive, philosophical |
| **Team Amber** | `en-IN-NeerjaNeural` | Indian English (Female) | Warm, engaging, articulate |
| **Team Silver** | `en-US-JennyNeural` | US English (Female) | Natural, conversational |

---

## 🛡️ 6. Acoustic Resilience Features

- **Half-Duplex Echo Lock:** A laptop never records from its microphone while its own speaker is active, plus a `0.35s` post-speech cooldown.
- **Adaptive Ambient Noise Calibration:** Every agent samples 1 second of ambient room noise on startup to calibrate its VAD energy threshold automatically.
- **Homophone-Tolerant Wake Detection:** Handles STT variations (`"team read"` ➔ `"Team Red"`, `"team blew"` ➔ `"Team Blue"`, `"mike"` ➔ `"mic"`).
- **Spoken Wake Enforcement:** Even if an LLM forgets to include the wake phrase in its text, `parse_agent_turn()` automatically appends `"Passing the mic to <Next Team>!"` before TTS synthesis so the next laptop's microphone is guaranteed to hear the trigger.

---

## 📜 License

MIT License. Crafted for live, in-person multi-agent voice AI meetups and hackathons.
