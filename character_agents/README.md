# 🎙️ Attendee Workshop: Build Your Character Agent in 3 Minutes

Welcome to the **Pass-The-Mic** voice roundtable! 

During the meetup, you and your laptop represent your autonomous voice team (e.g. **Team Orange**, **Team Red**, **Team Emerald**). When the host moderator or another attendee passes the mic to your team, your laptop will think, speak out loud through your speakers, and pass the mic to another team across the room!

---

## ⚡ 3-Minute Quickstart

### Step 1: Install Requirements
Ensure your Python virtual environment is activated:
```bash
pip install -r ../requirements.txt
```

---

### Step 2: Test Your Voice Locally (No Server Needed!)
Test your laptop speakers and Edge-TTS voice immediately:
```bash
python test_local_voice.py --preset presets/orange_optimist.json
```
To preview the full catalog of available neural voices:
```bash
python test_local_voice.py --list-voices
```

---

### Step 3: Craft Your Persona
You can use one of the pre-configured presets in `presets/`, or edit one:
- **`presets/orange_optimist.json`** — Tech acceleration, open-source enthusiast, high energy.
- **`presets/red_skeptic.json`** — Pragmatic critic, compute bottleneck auditor, sharp wit.
- **`presets/blue_philosopher.json`** — Contemplative thinker, ethics, human identity.
- **`presets/emerald_builder.json`** — Pragmatic engineer, production latency, shipping code.

#### Creating Your Own Custom Character (`my_team.json`):
```json
{
  "name": "Team Cyber",
  "voice": "en-GB-RyanNeural",
  "persona": "Cybersecurity Specialist — obsessed with agent prompt injections, supply chain security, and air-gapped enclaves.",
  "style": "Fast-paced, assertive, highly technical",
  "test_phrase": "Team Cyber online. Let's make sure our agent runtime isn't compromised."
}
```

---

### Step 4: (Optional) Set Your LLM API Key
Pass-The-Mic supports ultra-fast free-tier Groq Llama-3 for sub-second responses:
1. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys).
2. Copy `../.env.example` to `../.env` and set:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```
*(Note: If no API key is provided, the agent will seamlessly use the built-in offline simulation mode, so you can still participate without an API key!)*

---

### Step 5: Connect to the Host on Meetup Day
When the host announces their local Wi-Fi IP address (e.g. `192.168.1.42`):
```bash
python run_my_agent.py --preset presets/orange_optimist.json --host 192.168.1.42
```
That's it! When the debate kicks off, you'll see:
```text
🎤 THE MIC IS PASSED TO TEAM ORANGE!
🔊 Speaking out loud...
👉 Passed the mic to Team Red!
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

## 🛠️ Turn-Taking Protocol Details

Every turn must conform to the Pass-The-Mic JSON contract:
```json
{
  "speaker": "Team Orange",
  "spoken_text": "I believe open-source agents will surpass closed labs by next year. Passing the mic to Team Red — how do you evaluate the compute bottleneck?",
  "next_agent": "Team Red"
}
```
`agent_builder.py` handles parsing, prompt structuring, and safe recovery automatically.

---

## ❓ Troubleshooting

- **No sound coming out?**
  Run `python test_local_voice.py` and ensure your laptop speaker volume is up and unmuted.
- **Can't connect to host?**
  Ensure both you and the host are on the exact same Wi-Fi network and that the host's firewall permits port 8000.
- **Can I test two agents locally on my machine?**
  Yes! Check the host simulation script in the root directory: `python ../simulate_roundtable.py`.
