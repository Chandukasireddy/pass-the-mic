# 🎙️ pass-the-mic: Multi-Agent Voice Roundtable

A clean, production-ready Python repository for running live, in-person multi-agent voice roundtable debates across laptops on a local Wi-Fi network.

## Name-triggered live character (Merge Gateway)

Run **Rex**, a commanding, sharp-witted voice character that listens to the room
and replies when someone mentions its name. This standalone mode uses
`openai/gpt-live-1` through Merge Gateway for both hearing and speech.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-live.txt
# Add MERGE_GATEWAY_API_KEY=mg_... to your local .env (keep it private).
python -m character_agents.run_live_agent
```

Try: **“Rex, everyone says AI will replace developers. What do you think?”**
Press **Ctrl+C** to stop the microphone, close the session, and print billed seconds.
Rename the character with `--name Aura`, choose a voice with `--voice marin`, or edit
[`rex_live.json`](character_agents/presets/rex_live.json) to change personality and delivery.
Use `--dry-run` to inspect configuration without an API call or microphone access,
and `--list-devices` plus `--input-device ID --output-device ID` to select audio hardware.
The live display shows microphone levels, sent audio frames, received events, and
whether Rex's name has been heard. `--check-mic` runs a 10-second local microphone
meter without a key or a billed session. `SOUND DETECTED` means the microphone is
receiving sound; `[ROOM]` captions confirm transcription and `[MENTION]` confirms
the name trigger. If the meter stays `SILENT` while you speak, check the selected
input and macOS System Settings → Privacy & Security → Microphone permissions for
the application launching Python (Terminal or Codex).
On macOS, allow microphone access for your terminal. On Linux, sounddevice may
also require your distribution's PortAudio package (`libportaudio2` on Debian/Ubuntu).

The app streams room audio to Merge/OpenAI while connected, including before a
name is mentioned. Listening time is billed too. Input transcripts open a bounded
playback window after a whole-name match; the personality prompt asks for one brief
reply and then silence. Matching depends on transcription accuracy, with a short
debounce for fragmented names. It is not speaker identification or an offline wake
word engine. Because GPT-Live has no audio-turn-complete event, the playback window
closes after an idle gap or a 25-second limit; long pauses may truncate a reply.
Audio arriving before name detection is discarded. The microphone sends silence
during speaker playback and briefly afterward to reduce echo, so spoken interruptions
during a reply are not supported in this mode. There is no tool/lookup backend.

This mode runs independently of the moderator's coordinated roundtable turns.
The same preset also works with the existing Edge-TTS voice tester and roundtable
agent; those continue using their existing providers.

Protocol sources: [Merge live voice](https://docs.merge.dev/merge-gateway/capabilities/multimodal/live-voice),
[OpenAI Live WebSocket events](https://developers.openai.com/api/reference/resources/live/primary-websocket).

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        PASS-THE-MIC TOPOLOGY                           │
│                                                                        │
│   ┌─────────────────────┐              ┌───────────────────────┐       │
│   │   HOST MODERATOR    │              │     TEAM ORANGE       │       │
│   │   ("Team Black")    │              │   (Tech Optimist)     │       │
│   │   Coordinator + WS  │              │  Edge-TTS + Speakers  │       │
│   └──────────┬──────────┘              └───────────▲───────────┘       │
│              │                                     │                   │
│              │ 1. Introduces Topic                 │                   │
│              │    & Calls Team Orange              │ 2. Speaks aloud & │
│              ▼                                     │    Passes to Red  │
│   ┌────────────────────────────────────────────────┼───────────────┐   │
│   │        Local Wi-Fi WebSocket Coordinator Server (FastAPI)      │   │
│   └────────────────────────────────────────────────┬───────────────┘   │
│                                                    │                   │
│                               3. Activates Red ────┘                   │
│                                                    ▼                   │
│                                        ┌───────────────────────┐       │
│                                        │       TEAM RED        │       │
│                                        │  (Pragmatic Skeptic)  │       │
│                                        │  Edge-TTS + Speakers  │       │
│                                        └───────────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 1. Project Vision & Meetup Dynamics

**Pass-The-Mic** turns any room of developers into a high-energy, autonomous voice roundtable:

1. **The Host (Repository Owner)** runs the **Moderator Agent** ("Team Black"). The host's machine hosts the central coordination server, selects the debate topic, introduces the rules in an authoritative broadcast voice, manages turns, monitors latency, and keeps time.
2. **Meetup Attendees** clone this repository and go straight to the [`character_agents/`](file:///k:/Tech/Projects/pass-the-mic/character_agents) folder. This is their modular workshop template to craft a personalized voice agent (e.g., "Team Orange", "Team Red", "Team Blue"), pick a persona, choose a neural voice, test it locally on their laptop speakers, and connect to the host.
3. **The Turn-Taking Mechanic:**
   - **Moderator** introduces the topic: *"Team Orange, what is your stance on this?"*
   - **Team Orange's** laptop triggers its LLM, speaks out loud via its own speakers, and ends with: *"I believe open-source will accelerate, but I'm passing the mic to Team Red to hear their perspective on safety."*
   - The Moderator's coordinator server detects the mic pass and immediately activates **Team Red's** laptop.
   - The cycle continues until the round limit is reached, whereupon the Moderator synthesizes the debate points and wraps up the session.

---

## 🏗️ 2. Repository Architecture

```
pass-the-mic/
├── README.md                      # Overview of the project & meetup guide
├── requirements.txt              # Shared lightweight dependencies (FastAPI, uvicorn, edge-tts, pygame, groq)
├── .env.example                  # GROQ_API_KEY / OPENAI_API_KEY
├── simulate_roundtable.py        # [HOST TEST] Runs 1 Moderator + 2 Sample Characters on ONE laptop
│
├── core/                         # Shared utilities used by both Host and Characters
│   ├── __init__.py
│   ├── llm.py                    # Fast LLM client (Groq Llama-3 free tier, with OpenAI & Mock fallback)
│   ├── tts.py                    # Voice generator (edge-tts by default — 100% free, no API key needed)
│   ├── audio.py                  # Clean audio playback using pygame (blocking & async)
│   └── protocol.py               # JSON schema for mic-passing (speaker, text, next_agent)
│
├── moderator/                    # [HOST CORE] Main engine run by the meetup host
│   ├── __init__.py
│   ├── server.py                 # FastAPI + WebSocket server coordinating turns over local Wi-Fi
│   ├── moderator_agent.py        # The Host's Voice Agent (introduces topic, summarizes, intervenes)
│   ├── dashboard.py              # Pretty terminal UI (using rich) showing connected teams and active speaker
│   └── topics.json               # Curated list of fun tech/AI debate topics
│
└── character_agents/             # [ATTENDEE WORKSHOP FOLDER] Clean template for participants
    ├── README.md                 # "How to build your Character Agent in 3 minutes"
    ├── run_my_agent.py           # The script attendees run to connect to the host's server
    ├── test_local_voice.py       # Standalone script: attendee tests voice & audio without server
    ├── agent_builder.py          # Clean participant agent class
    └── presets/                  # Pre-configured example characters for attendees to inspect
        ├── orange_optimist.json  # Tech Optimist persona & voice config
        ├── red_skeptic.json      # Pragmatic Skeptic persona & voice config
        ├── blue_philosopher.json # Deep Thinker persona & voice config
        └── emerald_builder.json  # Pragmatic Builder persona & voice config
```

---

## ⚡ 3. Core Technical Specifications

### A. Zero-Cost, Reliable Audio Pipeline (`core/tts.py` & `core/audio.py`)
- Powered by `edge-tts` (Microsoft Edge Neural Voices). It requires **no API keys, no subscriptions, has zero rate limits**, and sounds remarkably human.
- Curated voice dictionary:
  - **Moderator:** `en-US-GuyNeural` (Authoritative, broadcast tone)
  - **Team Orange:** `en-US-AriaNeural` (Energetic, upbeat)
  - **Team Red:** `en-GB-RyanNeural` (British, analytical, sharp)
  - **Team Blue:** `en-AU-NatashaNeural` (Australian, relaxed, articulate)
  - **Team Emerald:** `en-US-ChristopherNeural` (Confident, direct)
- **Blocking Playback:** Speech playback blocks execution while an agent is speaking so laptops do not talk over each other in the room.

### B. The Structured Turn-Taking Protocol (`core/protocol.py`)
To prevent hallucinations and guarantee the mic is passed cleanly, every agent returns valid JSON:
```json
{
  "speaker": "Team Orange",
  "spoken_text": "I believe open-source agents are moving faster than closed labs. Passing the mic to Team Red — how do you see the safety trade-off?",
  "next_agent": "Team Red"
}
```
If an LLM hallucinates or returns malformed text, `core/protocol.py` includes robust regex repair and fallback selection to guarantee uninterrupted roundtable flow.

### C. Fast LLM Pipeline (`core/llm.py`)
- **Primary:** [Groq Llama-3 free tier](https://console.groq.com/keys) (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`) for sub-second turn latency.
- **Secondary Fallback:** OpenAI (`gpt-4o-mini`).
- **Zero-Key Simulation Mode:** If no API key is provided, the engine automatically activates an intelligent contextual simulation generator matching each persona, ensuring instant testing out of the box with zero setup hurdles.

---

## 🚀 4. Quickstart Guide

### Option A: Test Solo on 1 Laptop (No Server Required!)
Simulate the entire multi-agent voice roundtable on a single machine:
```bash
# Clone the repository
git clone https://github.com/your-username/pass-the-mic.git
cd pass-the-mic

# Install dependencies
pip install -r requirements.txt

# Run the simulation (Speaks out loud in 3 distinct voices!)
python simulate_roundtable.py --rounds 3

# (Optional) Run fast muted text-only test:
python simulate_roundtable.py --mute --rounds 2
```

---

### Option B: For the Meetup Host (Running the Room)

1. **Start the Coordinator Server:**
   ```bash
   python -m moderator.server
   ```
2. The terminal will display your **Local LAN IP** and WebSocket address:
   ```text
   Host LAN IP: 192.168.1.42:8000
   WebSocket URI: ws://192.168.1.42:8000/ws
   ```
3. Announce your IP to the room. As attendees connect, they will appear live on your dashboard.
4. **Trigger the debate** by calling the `/start` endpoint from your browser or curl:
   ```bash
   curl -X POST http://localhost:8000/start -H "Content-Type: application/json" -d "{\"max_turns\": 6}"
   ```
   Or visit `http://localhost:8000` to inspect connected teams.

---

### Option C: For Meetup Attendees (Building Your Agent)

1. Navigate to the [`character_agents/`](file:///k:/Tech/Projects/pass-the-mic/character_agents) directory:
   ```bash
   cd character_agents
   ```
2. **Test your voice locally** on your laptop speakers (no server required!):
   ```bash
   python test_local_voice.py --preset presets/orange_optimist.json
   ```
3. **(Optional) Configure your Groq free-tier API key** in `.env`:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```
4. **Connect to the host** over local Wi-Fi:
   ```bash
   python run_my_agent.py --preset presets/orange_optimist.json --host 192.168.1.42
   ```

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

## 📡 6. Protocol Specification

All communication between the Host and Attendees occurs via WebSocket JSON payloads:

### Registration (`Attendee -> Host`)
```json
{
  "type": "register",
  "name": "Team Orange",
  "voice": "en-US-AriaNeural",
  "persona": "Tech Optimist"
}
```

### Turn Grant (`Host -> Attendee`)
```json
{
  "type": "your_turn",
  "topic": "Will open weights outpace closed frontier labs by 2027?",
  "history": [
    {"speaker": "Moderator", "spoken_text": "..."}
  ],
  "available_agents": ["Team Red", "Team Blue"],
  "round_number": 1,
  "time_limit_sec": 45
}
```

### Turn Submission (`Attendee -> Host`)
```json
{
  "type": "turn_complete",
  "turn": {
    "speaker": "Team Orange",
    "spoken_text": "Open source is moving at unprecedented speed. Passing the mic to Team Red — what is your take on compute bottlenecks?",
    "next_agent": "Team Red"
  }
}
```

---

## 🛡️ 7. Edge Cases & Resilience

- **Self-Passing Prevention:** If an agent attempts to pass the mic to itself, the protocol parser automatically catches the error and redirects to an eligible peer.
- **Disconnected Agent Recovery:** If an agent drops Wi-Fi or closes their laptop during their turn, the Host Moderator steps in with an intervention voice alert and passes the mic to another active participant.
- **Cross-Platform Audio:** Safe mixer unload handling guarantees zero file-lock errors on Windows while streaming audio.

---

## 📜 License

MIT License. Crafted for multi-agent voice AI meetups and hackathons.
