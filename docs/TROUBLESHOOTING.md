# Troubleshooting

Every issue here was encountered and solved during real testing on MacBook Pro M4 Pro 24GB.

## Training Errors

### `RuntimeError: expected scalar type Float but found Half`

**Where:** Training (2-get-hubert-wav32k.py), Inference (inference_webui.py), API (TTS.py)

**Cause:** GPT-SoVITS loads HuBERT model weights stored in float16 format. Mac CPU/MPS can't do float16 convolutions.

**Fix:** Run `python patches/apply_mac_fixes.py` — this replaces all `.half()` with `.float()` and forces float32 everywhere.

If patches don't apply cleanly, manually add `.float()` after every model load:
```python
model = model.float().to(device)
tensor = tensor.float().to(device)
```

### `LookupError: Resource 'averaged_perceptron_tagger_eng' not found`

```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

### `ImportError: TorchCodec is required`

Demucs completes processing but crashes when saving output.

```bash
pip install torchcodec
```

Or skip denoising if audio is clean: `make process -- --skip-denoise`

### `ImportError: cannot import name 'HfFolder'`

```bash
pip install --upgrade gradio huggingface_hub
```

### `ImportError: cannot import name 'HybridCache'`

```bash
pip install --upgrade transformers peft
```

### `FileNotFoundError: logs/xxx/6-name2semantic.tsv`

GPT training can't find the experiment data. Fix:
```bash
sed -i '' 's|logs/xxx/|logs/voice_clone/|g' GPT-SoVITS/TEMP/tmp_s1.yaml
```

### GPU shows "0 CPU" — should I change it?

No. The GPU selector is NVIDIA CUDA only. Mac uses CPU with Metal optimizations internally.

## Inference Errors

### `OSError: Reference audio is outside the 3-10 second range`

Your reference clip is too short or too long. Find a 4-8 second clip:
```bash
for f in sliced/seg_00{30..50}.wav; do
  echo "$f: $(soxi -D $f 2>/dev/null)s"
done
```

### Generated audio is only 1-2 seconds / cuts off early

**If using Gradio API:** The `get_tts_wav` endpoint is a Python generator that yields chunks. `gradio_client.predict()` only captures the first chunk.

**Fix:** Use the REST API (`api_v2.py` on port 9880) instead of Gradio.

**If using REST API:** The model hits EOS (end of sequence) too early.
- Set `repetition_penalty: 1.35`
- Set `temperature: 0.6` (not lower — too low triggers early stopping)
- Use `text_split_method: "cut5"`

### Voice quality is poor / robotic

1. Lower temperature to 0.6
2. Try different reference clips — this has the biggest impact
3. Try earlier checkpoints (e8 sometimes > e14)
4. Keep sentences short for inference
5. Use training data with natural conversational speech

## API Errors

### `Connection refused` on port 9872

The inference WebUI isn't started. Open `http://localhost:9874`, go to 1C tab, select models, click "Open TTS Inference WebUI".

### `Connection refused` on port 9880

The REST API isn't running:
```bash
cd GPT-SoVITS && python3 api_v2.py -a 0.0.0.0 -p 9880
```

### LLM generates too much text / role-plays caller

Add stop sequences:
```python
"options": {"stop": ["Caller:", "Customer:", "\n\n"]}
```

And limit output: `"num_predict": 40`

## Performance

| Optimization | Effect |
|---|---|
| Whisper tiny vs base | 400ms saved |
| phi3:mini vs mistral | 500ms saved |
| `keep_alive: 24h` in Ollama | Eliminates cold start |
| Short sentences to TTS | Better quality per sentence |
| Right reference clip | Single biggest quality factor |
