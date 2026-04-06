# Training Guide

Step-by-step guide to train your voice model using GPT-SoVITS WebUI.

## Prerequisites

- Ran `make process` successfully (you should have 100+ segments in `sliced/`)
- `dataset/transcription.list` exists with transcriptions

## Step 1: Launch WebUI

```bash
make train
# Opens http://localhost:9874
```

## Step 2: Set Experiment Name

At the top of the page:
- **Experiment name:** `voice_clone` (or any name)
- **GPU Information:** Leave as `0 CPU` (this is for NVIDIA only, Mac uses CPU)
- **Version:** Select `v2Pro`

## Step 3: Dataset Formatting (1A tab)

Fill in:
- **Text labelling file:** Full path to `dataset/transcription.list`
- **Audio dataset folder:** Full path to `sliced/`

Click **"Open Training Set One-Click Formatting"** and wait. This runs:
1. Tokenization & BERT feature extraction
2. HuBERT SSL feature extraction  
3. Semantic token extraction

Should complete in 5-10 minutes. You'll see "Training Set One-Click Formatting Finished".

## Step 4: Fine-Tuning (1B tab)

### SoVITS Training
- **Batch size:** 10 (M4 Pro 24GB) or 5 (8GB Mac)
- **Total epochs:** 14
- **Save every:** 2 epochs
- Click **"Start SoVITS Training"**
- Wait ~30-45 minutes on M4 Pro

### GPT Training
- **Batch size:** 6
- **Total epochs:** 22
- **Save every:** 4 epochs
- Click **"Start GPT Training"**
- Wait ~30-45 minutes on M4 Pro

## Step 5: Test (1C tab)

1. Click **"refreshing model paths"**
2. Select your GPT model (e.g., `voice_clone-e20.ckpt`)
3. Select your SoVITS model (e.g., `voice_clone_e14_s602.pth`)
4. Click **"Open TTS Inference WebUI"**
5. Upload a reference audio clip (3-10 seconds)
6. Enter the reference text
7. Enter text to synthesize
8. Click **"Start inference"**

### Settings for Best Quality
- **temperature:** 0.6
- **top_k:** 10
- **top_p:** 1.0
- **How to slice:** "Slice by English punct"
- **Speech rate:** 1.0

## Step 6: Start the API

For the voice agents, use the REST API instead of the WebUI:

```bash
cd GPT-SoVITS
python3 api_v2.py -a 0.0.0.0 -p 9880
```

Then set your trained weights:
```bash
curl "http://localhost:9880/set_sovits_weights?weights_path=SoVITS_weights_v2Pro/voice_clone_e14_s602.pth"
curl "http://localhost:9880/set_gpt_weights?weights_path=GPT_weights_v2Pro/voice_clone-e20.ckpt"
```

## Tips

- **Reference clip is critical.** Try 5-6 different clips and compare output quality.
- **Don't overtrain.** Compare e8 vs e12 vs e14 — sometimes earlier epochs sound more natural.
- **Conversational audio > scripted.** Natural speech patterns produce better general-purpose TTS.
- **Clean your transcriptions.** Fix Whisper errors in `transcription.list` before training.
