# 🎙️ Voice Clone Toolkit

**Clone any voice. Run it locally. Zero cloud costs.**

Train a custom TTS model from your own recordings and deploy it as a real-time voice agent — all on Apple Silicon.



> ⚡ 37 min of audio → trained in ~1 hour → 1.5s response latency → real-time voice conversations

---

## What This Does

1. **You record yourself** (or anyone who consents) speaking for 5-60 minutes
2. **The toolkit processes the audio** — denoises, slices, transcribes automatically
3. **Trains a voice model** using GPT-SoVITS (fine-tuning, not zero-shot)
4. **You get an API** that generates speech in that voice from any text
5. **Bonus: real-time voice agent** — talk to an AI that responds in your cloned voice with ~1.5s latency

Everything runs locally on your Mac. No API keys, no cloud, no monthly fees.

## Demo

```
You: "What is the capital of India?"

🤖 Agent (in YOUR voice): "The capital of India is New Delhi."

Total latency: 1,557ms
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/voice-clone-toolkit.git
cd voice-clone-toolkit

# One-time setup (~15 min)
make setup

# Place your WAV recording
cp /path/to/your/recording.wav raw/

# Process audio → training segments
make process

# Train via WebUI (opens browser)
make train

# Real-time voice chat
make chat
```

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **Hardware** | Apple Silicon Mac (M1+) | M4 Pro 24GB |
| **RAM** | 8GB | 16-24GB |
| **Storage** | 15GB free | 25GB free |
| **Python** | 3.10 | 3.11 |
| **Audio** | 5 min WAV | 30-60 min WAV |
| **OS** | macOS 13+ | macOS 14+ |

> Also works on Linux with NVIDIA GPU. See [docs/SETUP_LINUX.md](docs/SETUP_LINUX.md).

## How It Works

```
your_recording.wav (5-60 min)
    │
    ├── [01_process_audio.py] ──── Denoise → Slice → Transcribe
    │                                    ↓
    │                              204 training segments
    │                                    ↓
    ├── [GPT-SoVITS WebUI] ────── Feature extraction → SoVITS training → GPT training
    │                                    ↓
    │                              Trained voice model
    │                                    ↓
    ├── [REST API] ────────────── Text in → Speech out (WAV)
    │                                    ↓
    └── [Real-time Agent] ─────── You speak → AI thinks → Your voice responds
                                        (~1.5 second latency)
```

## Training Data Guidelines

| Audio Length | Voice Quality | Training Time (M4 Pro) |
|---|---|---|
| 1-5 min | Basic | ~15 min |
| 10-20 min | Good | ~30 min |
| **30-60 min** | **Excellent** | **~1 hour** |
| 2+ hours | Diminishing returns | 3+ hours |

**Best results:**
- Single speaker, clean audio, minimal background noise
- Natural conversational speech > scripted narration
- Consistent recording quality (same mic, same session)
- WAV format, any sample rate

## Deployment Options

### 1. Real-Time Voice Agent (via microphone)
```bash
make chat
# Speak → AI responds in your cloned voice
# ~1.5s latency on M4 Pro
```

### 2. REST API
```bash
make api
# Then:
curl -X POST http://localhost:9880/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "text_lang": "en", ...}' \
  --output speech.wav
```

### 3. Web Support Agent
```bash
make support
# Opens http://localhost:7861
# Type questions, hear responses in cloned voice
```

### 4. Phone Agent (Twilio)
```bash
make phone
# Requires: Twilio account + ngrok
```

## Performance

Benchmarked on MacBook Pro M4 Pro 24GB:

| Component | Latency |
|---|---|
| Speech-to-Text (Whisper tiny) | ~200ms |
| LLM (phi3:mini) | ~50-100ms |
| Text-to-Speech (GPT-SoVITS) | ~1,100ms |
| **Total (first audio)** | **~1,500ms** |

## Apple Silicon Compatibility

This toolkit auto-patches GPT-SoVITS for Mac. Without these patches, you'll hit `expected Float but found Half` errors on every step. The patches are applied automatically during `make setup`.

| Issue | Status |
|---|---|
| HuBERT float16 → float32 | ✅ Auto-fixed |
| Inference float16 → float32 | ✅ Auto-fixed |
| API float16 → float32 | ✅ Auto-fixed |
| NLTK missing data | ✅ Auto-fixed |
| torchcodec missing | ✅ Auto-fixed |
| Gradio/HuggingFace version conflicts | ✅ Documented |

## vs ElevenLabs

| Feature | ElevenLabs | This Toolkit |
|---|---|---|
| Monthly cost | $5-50/month | Free |
| Voice quality | 9/10 | 8/10 |
| Data privacy | Their servers | 100% local |
| Fine-tuning control | Limited | Full |
| Latency (real-time) | ~1s | ~1.5s |
| Mac native | No | Yes |
| Languages | 30+ | 6 (en, zh, ja, ko, yue, mixed) |
| Requires internet | Yes | No |

## Project Structure

```
voice-clone-toolkit/
├── Makefile                    # make setup / process / train / chat
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── pipeline/
│   ├── 01_process_audio.py     # Audio → training segments
│   ├── 02_voice_api.py         # REST API for TTS
│   └── config.yaml             # Default settings
│
├── agents/
│   ├── realtime_agent.py       # Real-time voice chat (main demo)
│   ├── support_agent.py        # Web-based support agent
│   ├── phone_agent.py          # Twilio phone agent
│   └── local_voice_chat.py     # Simple mic-based chat
│
├── patches/
│   └── apply_mac_fixes.py      # Auto-fix GPT-SoVITS for Apple Silicon
│
├── docs/
│   ├── SETUP_MAC.md
│   ├── SETUP_LINUX.md
│   ├── TRAINING_GUIDE.md
│   └── TROUBLESHOOTING.md
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── raw/                        # Place your WAV file here
    └── .gitkeep
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — every error documented here was encountered and solved during real testing on M4 Pro.

## Credits

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — TTS engine
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Speech-to-text
- [Demucs](https://github.com/facebookresearch/demucs) — Audio source separation
- [Ollama](https://ollama.com) — Local LLM inference

## License

MIT — use for anything, including commercial. 

**Ethics:** Always get explicit consent before cloning someone's voice. Never use cloned voices to impersonate or deceive.

---

Built by [@devopswithakhil](https://instagram.com/devopswithakhil)
