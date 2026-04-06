#!/usr/bin/env python3
"""
Simple voice chat — no streaming, easy to understand.
Good starting point before using the full realtime_agent.

Usage: python local_voice_chat.py
"""

import os, sys, time, subprocess, re, uuid
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SLICED_DIR = PROJECT_DIR / "sliced"
OUTPUT_DIR = PROJECT_DIR / "output" / "chat"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_AUDIO = os.environ.get("REF_AUDIO", str(SLICED_DIR / "seg_0001.wav"))
REF_TEXT = os.environ.get("REF_TEXT",
    "part 11 windows containers containers are not just linux windows supports two types of containers process containers share the"
)


def record(path, sr=16000):
    print("\n🎤 Recording... (press Enter to stop)")
    p = subprocess.Popen(["rec", "-q", "-r", str(sr), "-c", "1", "-b", "16", path],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    input()
    p.terminate()
    p.wait()


def transcribe(path):
    from faster_whisper import WhisperModel
    m = WhisperModel("tiny", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(path, language="en")
    return " ".join(s.text.strip() for s in segs)


def ask_llm(question):
    import requests
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3:mini",
            "prompt": f"Answer in one sentence max 15 words. No markdown.\n\nQuestion: {question}\n\nAnswer:",
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 40, "stop": ["\n\n"]}
        }, timeout=15)
        return re.sub(r'[*#_`~]', '', r.json().get("response", "")).strip()
    except:
        return "Let me look into that for you."


def speak(text):
    import requests
    path = str(OUTPUT_DIR / f"resp_{uuid.uuid4().hex[:8]}.wav")
    try:
        r = requests.post("http://localhost:9880/tts", json={
            "text": text, "text_lang": "en",
            "ref_audio_path": REF_AUDIO, "prompt_text": REF_TEXT, "prompt_lang": "en",
            "text_split_method": "cut5", "top_k": 10, "top_p": 1.0,
            "temperature": 0.6, "repetition_penalty": 1.35, "seed": -1,
        }, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(path, "wb") as f:
                f.write(r.content)
            subprocess.run(["afplay", path], check=False)
    except Exception as e:
        print(f"  TTS error: {e}")


def main():
    print("\n🎙️  Voice Chat (Simple)")
    print("=" * 40)
    print("Speak → AI responds in your voice")
    print("Type 'quit' to exit\n")

    speak("Hello! How can I help you?")

    while True:
        cmd = input("\nPress Enter to speak (or 'quit'): ").strip()
        if cmd.lower() == "quit":
            break

        tmp = str(OUTPUT_DIR / "input.wav")
        record(tmp)

        text = transcribe(tmp)
        print(f"\n👤 You: {text}")
        if not text.strip():
            print("  (Didn't catch that)")
            continue

        start = time.time()
        response = ask_llm(text)
        print(f"🤖 Agent: {response}")

        speak(response)
        print(f"  ({(time.time()-start)*1000:.0f}ms total)")


if __name__ == "__main__":
    main()
