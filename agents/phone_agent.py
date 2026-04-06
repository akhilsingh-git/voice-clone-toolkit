#!/usr/bin/env python3
"""
Phone Voice Agent — Receive calls via Twilio, respond in cloned voice.

Setup:
  1. Create Twilio account → get phone number
  2. export TWILIO_ACCOUNT_SID="..." TWILIO_AUTH_TOKEN="..."
  3. Start: python phone_agent.py
  4. Start ngrok: ngrok http 8765
  5. Set Twilio webhook: https://YOUR_URL/incoming-call

See docs/TRAINING_GUIDE.md for full instructions.
"""

import os, sys, json, time, base64, asyncio, uuid, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phone")

PROJECT_DIR = Path(__file__).parent.parent
SLICED_DIR = PROJECT_DIR / "sliced"
OUTPUT_DIR = PROJECT_DIR / "output" / "phone"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_AUDIO = os.environ.get("REF_AUDIO", str(SLICED_DIR / "seg_0001.wav"))
REF_TEXT = os.environ.get("REF_TEXT",
    "part 11 windows containers containers are not just linux windows supports two types of containers process containers share the"
)
PORT = 8765


def synthesize(text):
    import requests
    path = str(OUTPUT_DIR / f"tts_{uuid.uuid4().hex[:8]}.wav")
    try:
        r = requests.post("http://localhost:9880/tts", json={
            "text": text, "text_lang": "en",
            "ref_audio_path": REF_AUDIO, "prompt_text": REF_TEXT, "prompt_lang": "en",
            "text_split_method": "cut5", "top_k": 10, "top_p": 1.0,
            "temperature": 0.6, "repetition_penalty": 1.35,
        }, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
    except Exception as e:
        logger.error(f"TTS: {e}")
    return None


def create_app():
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import Response, HTMLResponse

    app = FastAPI()

    @app.post("/incoming-call")
    async def incoming(request: Request):
        host = request.headers.get("host", f"localhost:{PORT}")
        scheme = "wss" if "ngrok" in host else "ws"
        return Response(content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Say>Connecting you now.</Say>
<Connect><Stream url="{scheme}://{host}/media-stream" /></Connect></Response>""",
            media_type="application/xml")

    @app.websocket("/media-stream")
    async def media(ws: WebSocket):
        await ws.accept()
        logger.info("Call connected")
        # TODO: Implement real-time audio streaming
        # See agents/realtime_agent.py for the processing pipeline
        try:
            async for msg in ws.iter_text():
                data = json.loads(msg)
                if data.get("event") == "stop":
                    break
        except:
            pass
        logger.info("Call ended")

    @app.get("/test")
    async def test(text: str = "Hello, this is a test."):
        from fastapi.responses import FileResponse
        wav = synthesize(text)
        if wav:
            return FileResponse(wav, media_type="audio/wav")
        return {"error": "TTS failed"}

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return f"""<html><body style="font-family:sans-serif;background:#0d1117;color:#e6edf3;padding:60px">
<h1>📞 Phone Voice Agent</h1>
<p>1. <code>ngrok http {PORT}</code></p>
<p>2. Set Twilio webhook: <code>https://YOUR_URL/incoming-call</code></p>
<p>3. Call your Twilio number</p>
<p><br>Test TTS: <a href="/test?text=Hello+world" style="color:#ff872b">/test?text=Hello+world</a></p>
</body></html>"""

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"\n📞 Phone Agent → http://localhost:{PORT}")
    print(f"   Test: http://localhost:{PORT}/test?text=Hello+world\n")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
