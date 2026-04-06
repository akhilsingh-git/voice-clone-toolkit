# Linux Setup Guide (NVIDIA GPU)

With an NVIDIA GPU, training is faster and inference supports float16 natively.

## Prerequisites

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg sox git

# NVIDIA drivers + CUDA (if not already installed)
# See: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/voice-clone-toolkit.git
cd voice-clone-toolkit
make setup
```

The Mac patches will be skipped automatically on Linux since float16 works natively with CUDA.

## GPU Memory Requirements

| GPU VRAM | Max Batch Size | Training Speed |
|---|---|---|
| 4GB (GTX 1050) | 2-4 | Slow, may swap |
| 8GB (RTX 3060) | 8-10 | Good |
| 12GB+ (RTX 3060 12GB) | 12-16 | Fast |
| 24GB (RTX 4090) | 16-24 | Very fast |

## Using GPU

Set GPU in the WebUI dropdown instead of CPU. The toolkit auto-detects CUDA.
