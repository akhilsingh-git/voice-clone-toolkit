PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup process train chat api support phone clean help

help:
	@echo "🎙️  Voice Clone Toolkit"
	@echo ""
	@echo "  make setup     - Install everything (run once)"
	@echo "  make process   - Process your WAV file into training data"
	@echo "  make train     - Open GPT-SoVITS WebUI for training"
	@echo "  make chat      - Real-time voice chat (main demo)"
	@echo "  make api       - Start TTS REST API"
	@echo "  make support   - Start web support agent"
	@echo "  make phone     - Start phone agent (needs Twilio)"
	@echo "  make benchmark - Run latency benchmark"
	@echo "  make clean     - Remove generated files"

# ---- Setup ----
setup: venv deps gpt-sovits patches nltk-data dirs
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Place your WAV file in: raw/"
	@echo "  2. Run: make process"
	@echo "  3. Run: make train"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		$(PIP) install --upgrade pip wheel setuptools; \
	fi

deps: venv
	$(PIP) install -r requirements.txt
	@command -v brew >/dev/null && brew install ffmpeg sox 2>/dev/null || true

gpt-sovits: venv
	@if [ ! -d "GPT-SoVITS" ]; then \
		echo "📦 Cloning GPT-SoVITS..."; \
		git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git; \
		cd GPT-SoVITS && $(PIP) install -r requirements.txt; \
	fi

patches: gpt-sovits
	@echo "🩹 Applying Mac compatibility patches..."
	$(PY) patches/apply_mac_fixes.py

nltk-data: venv
	$(PY) -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('cmudict', quiet=True)"

dirs:
	@mkdir -p raw cleaned sliced dataset models output logs

# ---- Pipeline ----
process: venv
	$(PY) pipeline/01_process_audio.py

train: gpt-sovits
	@echo "Opening GPT-SoVITS WebUI..."
	@echo "See docs/TRAINING_GUIDE.md for step-by-step instructions."
	cd GPT-SoVITS && $(PY) webui.py

api: gpt-sovits
	@echo "Starting TTS REST API on port 9880..."
	cd GPT-SoVITS && $(PY) api_v2.py -a 0.0.0.0 -p 9880

# ---- Agents ----
chat: venv
	$(PY) agents/realtime_agent.py --stt-model tiny

support: venv
	$(PY) agents/support_agent.py

phone: venv
	$(PY) agents/phone_agent.py

benchmark: venv
	$(PY) agents/realtime_agent.py --benchmark

# ---- Cleanup ----
clean:
	rm -rf cleaned/ sliced/ dataset/ output/ logs/
	rm -rf GPT-SoVITS/logs/ GPT-SoVITS/TEMP/

clean-all: clean
	rm -rf $(VENV) GPT-SoVITS/
