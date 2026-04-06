#!/usr/bin/env python3
"""
Web Support Agent — Type questions, hear responses in cloned voice.

Prerequisites:
  1. GPT-SoVITS api_v2.py running on port 9880
  2. Ollama running

Usage:
  python support_agent.py
  # Open http://localhost:7861
"""

import os, sys, time, re, uuid, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support-agent")

PROJECT_DIR = Path(__file__).parent.parent
SLICED_DIR = PROJECT_DIR / "sliced"
OUTPUT_DIR = PROJECT_DIR / "output" / "support"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_AUDIO = os.environ.get("REF_AUDIO", str(SLICED_DIR / "seg_0001.wav"))
REF_TEXT = os.environ.get("REF_TEXT",
    "part 11 windows containers containers are not just linux windows supports two types of containers process containers share the"
)
TTS_PORT = 9880


def synthesize(text):
    import requests
    output_path = str(OUTPUT_DIR / f"{uuid.uuid4().hex[:12]}.wav")
    try:
        resp = requests.post(f"http://localhost:{TTS_PORT}/tts", json={
            "text": text, "text_lang": "en",
            "ref_audio_path": REF_AUDIO, "prompt_text": REF_TEXT, "prompt_lang": "en",
            "text_split_method": "cut5", "top_k": 10, "top_p": 1.0,
            "temperature": 0.6, "repetition_penalty": 1.35, "seed": -1,
        }, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return output_path
    except Exception as e:
        logger.error(f"TTS: {e}")
    return None


async def ask_llm(question, context="general"):
    import aiohttp
    prompt = f"""You are a friendly support agent. Keep responses under 3 sentences.
Plain English only, no code or markdown. NEVER mention being AI.
Context: {context} support.

Customer: {question}

Agent:"""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post("http://localhost:11434/api/generate", json={
                "model": "phi3:mini", "prompt": prompt, "stream": False,
                "options": {"temperature": 0.3, "num_predict": 60,
                           "stop": ["Caller:", "Customer:", "\n\n"]}
            }, timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()
                return re.sub(r'[*#_`~]', '', data.get("response", "")).strip()
    except:
        return "Thanks for reaching out! Let me look into that for you."


def create_app():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="Voice Support Agent")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class Req(BaseModel):
        question: str
        context: str = "DevOps and cloud infrastructure"

    @app.post("/api/support")
    async def support(req: Req):
        start = time.time()
        text = await ask_llm(req.question, req.context)
        audio = synthesize(text)
        return {
            "question": req.question, "response_text": text,
            "audio_url": f"/api/audio/{Path(audio).name}" if audio else None,
            "latency_ms": round((time.time()-start)*1000),
        }

    @app.get("/api/audio/{fn}")
    async def audio(fn: str):
        p = OUTPUT_DIR / fn
        if p.exists():
            return FileResponse(str(p), media_type="audio/wav")
        from fastapi import HTTPException
        raise HTTPException(404)

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return """<!DOCTYPE html><html><head><title>Voice Support Agent</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;display:flex;justify-content:center;align-items:center}.c{max-width:600px;width:100%;padding:40px 20px}h1{font-size:24px;margin-bottom:8px}.s{color:#8b949e;margin-bottom:32px}textarea{width:100%;padding:16px;border:1px solid #30363d;border-radius:8px;background:#161b22;color:#e6edf3;font-size:16px;resize:vertical;min-height:80px;margin-bottom:20px}textarea:focus{outline:none;border-color:#ff872b}button{width:100%;padding:14px;border:none;border-radius:8px;background:#ff872b;color:#fff;font-size:16px;font-weight:600;cursor:pointer}button:hover{background:#e6751f}button:disabled{background:#30363d;cursor:wait}.r{margin-top:24px;padding:20px;background:#161b22;border:1px solid #30363d;border-radius:8px;display:none}.rt{margin-bottom:16px;line-height:1.6}.m{color:#8b949e;font-size:13px;margin-top:12px}audio{width:100%;margin-top:12px}.ex{margin-top:32px}.ex h3{font-size:14px;color:#8b949e;margin-bottom:12px}.eb{display:block;width:100%;text-align:left;padding:10px 14px;margin-bottom:8px;background:#161b22;border:1px solid #30363d;border-radius:6px;color:#e6edf3;cursor:pointer;font-size:14px}.eb:hover{border-color:#ff872b}</style></head>
<body><div class="c"><h1>Voice Support Agent</h1><p class="s">Ask a question, hear the response in a cloned voice</p>
<textarea id="q" placeholder="Type your question...">How do I set up resource limits in Kubernetes?</textarea>
<button id="b" onclick="ask()">Ask Support Agent</button>
<div id="r" class="r"><div id="rt" class="rt"></div><audio id="a" controls></audio><div id="m" class="m"></div></div>
<div class="ex"><h3>Try these:</h3>
<button class="eb" onclick="document.getElementById('q').value=this.textContent">My deployment keeps crashing with OOM errors</button>
<button class="eb" onclick="document.getElementById('q').value=this.textContent">What are your pricing plans?</button>
<button class="eb" onclick="document.getElementById('q').value=this.textContent">How do I configure SSL certificates?</button>
<button class="eb" onclick="document.getElementById('q').value=this.textContent">I want to cancel my subscription</button></div></div>
<script>async function ask(){const b=document.getElementById('b'),q=document.getElementById('q').value;if(!q)return;b.disabled=true;b.textContent='Generating...';try{const r=await fetch('/api/support',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const d=await r.json();document.getElementById('rt').textContent=d.response_text;if(d.audio_url){document.getElementById('a').src=d.audio_url;document.getElementById('a').play()}document.getElementById('m').textContent='Latency: '+d.latency_ms+'ms';document.getElementById('r').style.display='block'}catch(e){alert(e.message)}b.disabled=false;b.textContent='Ask Support Agent'}</script></body></html>"""

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"\n🎙️  Voice Support Agent → http://localhost:7861\n")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)
