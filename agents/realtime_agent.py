#!/usr/bin/env python3
"""
Real-Time Voice Agent — Talk to an AI that responds in your cloned voice.

Prerequisites:
  1. GPT-SoVITS api_v2.py running on port 9880
  2. Ollama running with phi3:mini (or mistral)
  3. sox installed (brew install sox)

Usage:
  python realtime_agent.py                    # Interactive voice chat
  python realtime_agent.py --benchmark        # Latency benchmark
  python realtime_agent.py --llm-model mistral  # Use different LLM
"""

import os, sys, time, asyncio, logging, re, queue, threading, shutil, uuid
from pathlib import Path
from typing import AsyncGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("realtime")

PROJECT_DIR = Path(__file__).parent.parent
SLICED_DIR = PROJECT_DIR / "sliced"
OUTPUT_DIR = PROJECT_DIR / "output" / "realtime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURE THESE — use the ref clip that works best for you
# ============================================================
REF_AUDIO = os.environ.get("REF_AUDIO", str(SLICED_DIR / "seg_0001.wav"))
REF_TEXT = os.environ.get("REF_TEXT",
    "part 11 windows containers containers are not just linux windows supports two types of containers process containers share the"
)

STT_MODEL = "tiny"
LLM_MODEL = "phi3:mini"
TTS_PORT = 9880


# ============================================================
# 1. Speech-to-Text
# ============================================================

class StreamingSTT:
    def __init__(self, model_size="tiny"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info(f"STT ready (faster-whisper {model_size})")

    def transcribe(self, audio_path):
        start = time.time()
        segments, _ = self.model.transcribe(
            audio_path, language="en", vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        text = " ".join(s.text.strip() for s in segments)
        logger.info(f"STT: {(time.time()-start)*1000:.0f}ms → '{text[:60]}...'")
        return text


# ============================================================
# 2. Streaming LLM — yields complete sentences
# ============================================================

conversation_history = []

async def stream_llm_sentences(question):
    import aiohttp

    system_prompt = """You are a support agent on a phone call.
Rules:
- Answer in ONE sentence only. Maximum 15 words.
- Be direct. No filler phrases like "I appreciate your curiosity" or "Great question".
- Never add history, context, or follow-up questions unless asked.
- Example: "What is the capital of India?" → "The capital of India is New Delhi."
- That's it. One sentence. Stop."""

    history = "\n".join(conversation_history[-6:])
    buffer = ""
    sentence_end = re.compile(r'(?<=[.!?])\s+')

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": f"{system_prompt}\n\n{history}\nCaller: {question}\n\nYou:",
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 40,
                        "stop": ["Caller:", "Customer:", "\n\n", "Is there"]
                    }
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                start = time.time()
                async for line in resp.content:
                    data = line.decode().strip()
                    if not data:
                        continue
                    import json
                    try:
                        chunk = json.loads(data)
                    except:
                        continue

                    buffer += chunk.get("response", "")
                    sentences = sentence_end.split(buffer)

                    if len(sentences) > 1:
                        for sent in sentences[:-1]:
                            sent = re.sub(r'[*#_`~]', '', sent).strip()
                            if len(sent) > 3:
                                logger.info(f"LLM sentence ({(time.time()-start)*1000:.0f}ms): '{sent}'")
                                yield sent
                        buffer = sentences[-1]

                    if chunk.get("done"):
                        break

                buffer = re.sub(r'[*#_`~]', '', buffer).strip()
                if len(buffer) > 3:
                    yield buffer

    except Exception as e:
        logger.error(f"LLM error: {e}")
        yield "Let me look into that for you."


# ============================================================
# 3. TTS via GPT-SoVITS REST API (port 9880)
# ============================================================

class TTSEngine:
    def __init__(self):
        pass

    def initialize(self):
        import requests
        try:
            requests.get(f"http://localhost:{TTS_PORT}/tts", timeout=5)
        except requests.exceptions.ConnectionError:
            logger.error(f"GPT-SoVITS API not running on port {TTS_PORT}!")
            logger.error("Start it: cd GPT-SoVITS && python3 api_v2.py -a 0.0.0.0 -p 9880")
            sys.exit(1)
        except:
            pass
        logger.info(f"TTS ready (GPT-SoVITS REST API on port {TTS_PORT})")

    def synthesize(self, text):
        import requests

        output_path = str(OUTPUT_DIR / f"tts_{uuid.uuid4().hex[:8]}.wav")
        start = time.time()

        try:
            response = requests.post(
                f"http://localhost:{TTS_PORT}/tts",
                json={
                    "text": text,
                    "text_lang": "en",
                    "ref_audio_path": REF_AUDIO,
                    "prompt_text": REF_TEXT,
                    "prompt_lang": "en",
                    "text_split_method": "cut5",
                    "top_k": 10,
                    "top_p": 1.0,
                    "temperature": 0.6,
                    "speed_factor": 1.0,
                    "repetition_penalty": 1.35,
                    "seed": -1,
                },
                timeout=30
            )

            if response.status_code == 200 and len(response.content) > 1000:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"TTS: {(time.time()-start)*1000:.0f}ms → {Path(output_path).name}")
                return output_path
            else:
                logger.error(f"TTS error: {response.text[:200]}")
        except Exception as e:
            logger.error(f"TTS error: {e}")
        return None


# ============================================================
# 4. Pipeline
# ============================================================

class RealtimePipeline:
    def __init__(self):
        self.stt = StreamingSTT(model_size=STT_MODEL)
        self.tts = TTSEngine()
        self.audio_queue = queue.Queue()

    def initialize(self):
        self.tts.initialize()
        self.playback_thread = threading.Thread(target=self._player, daemon=True)
        self.playback_thread.start()

    def _player(self):
        import subprocess
        while True:
            path = self.audio_queue.get()
            if path is None:
                break
            try:
                if os.path.exists(path):
                    subprocess.run(["afplay", path], check=False, capture_output=True)
            except Exception as e:
                logger.error(f"Play error: {e}")

    async def process_turn(self, audio_path):
        global conversation_history
        total_start = time.time()

        caller_text = self.stt.transcribe(audio_path)
        if not caller_text.strip():
            return None, None

        print(f"\n👤 Caller: {caller_text}")
        conversation_history.append(f"Caller: {caller_text}")

        # Collect full LLM response
        full_response = []
        async for sentence in stream_llm_sentences(caller_text):
            full_response.append(sentence)
            print(f"🤖 Agent: {sentence}")

        response_text = " ".join(full_response)
        conversation_history.append(f"You: {response_text}")

        # Single TTS call with full response
        wav_path = self.tts.synthesize(response_text)
        first_audio_time = None

        if wav_path:
            first_audio_time = (time.time() - total_start) * 1000
            local_copy = str(OUTPUT_DIR / f"play_{int(time.time()*1000)}.wav")
            shutil.copy2(wav_path, local_copy)
            self.audio_queue.put(local_copy)

        total_time = (time.time() - total_start) * 1000
        if first_audio_time:
            logger.info(f"Total turn: {total_time:.0f}ms | First audio: {first_audio_time:.0f}ms")

        return caller_text, response_text


# ============================================================
# 5. Microphone recording
# ============================================================

def record_mic(output_path, sr=16000):
    import subprocess
    print("\n🎤 Press Enter to start speaking...")
    input()
    print("   Recording... (press Enter to stop)")
    proc = subprocess.Popen(
        ["rec", "-q", "-r", str(sr), "-c", "1", "-b", "16", output_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    input()
    proc.terminate()
    proc.wait()


async def local_mic_mode():
    pipeline = RealtimePipeline()
    pipeline.initialize()

    print()
    print("🎙️  Real-Time Voice Agent")
    print("=" * 50)
    print(f"   STT: faster-whisper ({STT_MODEL})")
    print(f"   LLM: {LLM_MODEL}")
    print(f"   TTS: GPT-SoVITS REST API (port {TTS_PORT})")
    print(f"   Ref: {Path(REF_AUDIO).name}")
    print()
    print("   Type 'quit' to exit")
    print("=" * 50)

    # Greeting
    print(f"\n🤖 Agent: Hello! How can I help you today?")
    wav = pipeline.tts.synthesize("Hello! How can I help you today?")
    if wav:
        local = str(OUTPUT_DIR / f"greeting_{int(time.time()*1000)}.wav")
        shutil.copy2(wav, local)
        pipeline.audio_queue.put(local)

    while True:
        tmp = str(OUTPUT_DIR / "mic_input.wav")
        cmd = input("\nPress Enter to speak (or 'quit'): ").strip()
        if cmd.lower() == "quit":
            break
        record_mic(tmp)
        result = await pipeline.process_turn(tmp)
        if not result[0]:
            print("   (Didn't catch that)")


# ============================================================
# 6. Benchmark
# ============================================================

async def run_benchmark():
    print("\n📊 Latency Benchmark")
    print("=" * 50)

    stt = StreamingSTT(model_size=STT_MODEL)
    start = time.time()
    stt.transcribe(REF_AUDIO)
    stt_time = (time.time() - start) * 1000
    print(f"   STT ({STT_MODEL}): {stt_time:.0f}ms")

    start = time.time()
    first = None
    async for s in stream_llm_sentences("How do I deploy to Kubernetes?"):
        if first is None:
            first = (time.time() - start) * 1000
        break
    print(f"   LLM first sentence ({LLM_MODEL}): {first:.0f}ms")

    tts = TTSEngine()
    tts.initialize()
    start = time.time()
    tts.synthesize("Hello, how can I help you today?")
    tts_time = (time.time() - start) * 1000
    print(f"   TTS (GPT-SoVITS): {tts_time:.0f}ms")

    total = stt_time + first + tts_time
    print(f"\n   Estimated latency: {total:.0f}ms")
    print(f"   Target: <2000ms")
    print(f"   {'✅ GOOD' if total < 2000 else '⚠️  Needs optimization'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--stt-model", default="tiny", choices=["tiny", "base", "small"])
    parser.add_argument("--llm-model", default="phi3:mini")
    parser.add_argument("--tts-port", type=int, default=9880)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    args = parser.parse_args()

    STT_MODEL = args.stt_model
    LLM_MODEL = args.llm_model
    TTS_PORT = args.tts_port
    if args.ref_audio:
        REF_AUDIO = args.ref_audio
    if args.ref_text:
        REF_TEXT = args.ref_text

    if args.benchmark:
        asyncio.run(run_benchmark())
    else:
        asyncio.run(local_mic_mode())
