# Mac Setup Guide (Apple Silicon)

Tested on: MacBook Pro M4 Pro 24GB, Mac Mini M2 8GB

## Prerequisites

```bash
# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.11
brew install python@3.11

# System audio tools
brew install ffmpeg sox

# Ollama (for voice agents)
brew install ollama
ollama pull phi3:mini
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/voice-clone-toolkit.git
cd voice-clone-toolkit
make setup
```

This takes ~15 minutes and:
1. Creates a Python virtual environment
2. Installs all dependencies
3. Clones GPT-SoVITS
4. Applies Apple Silicon compatibility patches
5. Downloads NLTK data

## Memory Requirements

| Task | RAM Used |
|---|---|
| Audio processing | ~4GB |
| Training (SoVITS) | ~8-10GB |
| Training (GPT) | ~6-8GB |
| Inference | ~4GB |
| Real-time agent (all components) | ~6-8GB |

**8GB Mac:** Works but tight. Close other apps during training. Use batch_size=5.

**16-24GB Mac:** Comfortable. Use batch_size=10-12.

## Known Limitations

- Training runs on CPU (no CUDA). ~1 hour for 30 min of audio.
- MPS (Metal) is used partially but not for all operations.
- First inference call is slow (~3s) due to model loading. Subsequent calls are ~1s.
