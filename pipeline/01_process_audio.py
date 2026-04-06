#!/usr/bin/env python3
"""
Process a WAV recording into GPT-SoVITS training segments.

Pipeline: Denoise → Normalize → Slice → Filter → Transcribe

Usage:
  python 01_process_audio.py
  python 01_process_audio.py --skip-denoise     # if audio is already clean
  python 01_process_audio.py --language hi       # for Hindi
  python 01_process_audio.py --silence-thresh -45
"""

import os, sys, json, argparse, shutil, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence

PROJECT_DIR = Path(__file__).parent.parent
RAW_DIR = PROJECT_DIR / "raw"
CLEANED_DIR = PROJECT_DIR / "cleaned"
SLICED_DIR = PROJECT_DIR / "sliced"
DATASET_DIR = PROJECT_DIR / "dataset"
LOGS_DIR = PROJECT_DIR / "logs"

MIN_SEG_SEC = 3
MAX_SEG_SEC = 15
IDEAL_SEG_SEC = 10
MIN_RMS_DB = -45
MAX_RMS_DB = -5
MIN_SPEECH_RATIO = 0.3


def find_wav_file(specific_path=None):
    if specific_path:
        p = Path(specific_path)
        if p.exists():
            return p
        print(f"❌ File not found: {specific_path}")
        sys.exit(1)

    all_files = []
    for ext in ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.m4a", "*.M4A"]:
        all_files.extend(RAW_DIR.glob(ext))

    if not all_files:
        print(f"❌ No audio files found in {RAW_DIR}/")
        sys.exit(1)
    if len(all_files) == 1:
        return all_files[0]

    print(f"Found {len(all_files)} files:")
    for i, f in enumerate(all_files):
        print(f"  [{i+1}] {f.name}")
    choice = input("Which file? [1]: ").strip() or "1"
    return all_files[int(choice) - 1]


def get_duration_min(path):
    try:
        return sf.info(str(path)).duration / 60
    except:
        return len(AudioSegment.from_file(str(path))) / 60000


def denoise(input_path, skip=False):
    if skip:
        print("⏭️  Skipping denoising")
        out = CLEANED_DIR / "vocals.wav"
        CLEANED_DIR.mkdir(exist_ok=True)
        shutil.copy2(input_path, out)
        return out

    print(f"\n🔇 Step 1: Denoising with Demucs...")
    CLEANED_DIR.mkdir(exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals", "--out", str(CLEANED_DIR), str(input_path)
    ], check=True)

    for p in CLEANED_DIR.rglob("vocals.wav"):
        out = CLEANED_DIR / "vocals.wav"
        shutil.copy2(p, out)
        print(f"   ✅ Cleaned: {out}")
        return out

    print("   ⚠️ Demucs output not found, using original")
    out = CLEANED_DIR / "vocals.wav"
    shutil.copy2(input_path, out)
    return out


def normalize(input_path):
    print(f"\n🎚️  Step 2: Normalizing...")
    audio, sr = librosa.load(str(input_path), sr=None, mono=True)

    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        sr = 24000

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (10 ** (-1 / 20) / peak)

    from scipy.signal import butter, sosfilt
    sos = butter(5, 80, btype='highpass', fs=sr, output='sos')
    audio = sosfilt(sos, audio)

    out = CLEANED_DIR / "normalized.wav"
    sf.write(str(out), audio, sr)
    print(f"   ✅ {len(audio)/sr:.0f}s, {sr}Hz")
    return out


def slice_audio(input_path, silence_thresh=-38):
    print(f"\n✂️  Step 3: Slicing...")
    SLICED_DIR.mkdir(exist_ok=True)
    for f in SLICED_DIR.glob("*.wav"):
        f.unlink()

    audio = AudioSegment.from_file(str(input_path))
    chunks = split_on_silence(audio, min_silence_len=400, silence_thresh=silence_thresh, keep_silence=200)

    segments = []
    buffer = AudioSegment.empty()

    for chunk in chunks:
        chunk_d = len(chunk) / 1000
        buf_d = len(buffer) / 1000

        if chunk_d > MAX_SEG_SEC:
            if buf_d >= MIN_SEG_SEC:
                segments.append(buffer)
                buffer = AudioSegment.empty()
            for start in range(0, len(chunk), int(IDEAL_SEG_SEC * 1000)):
                piece = chunk[start:start + int(IDEAL_SEG_SEC * 1000)]
                if len(piece) / 1000 >= MIN_SEG_SEC:
                    segments.append(piece)
        elif buf_d + chunk_d <= MAX_SEG_SEC:
            buffer = buffer + chunk
        else:
            if buf_d >= MIN_SEG_SEC:
                segments.append(buffer)
            buffer = chunk

    if len(buffer) / 1000 >= MIN_SEG_SEC:
        segments.append(buffer)

    total = 0
    for i, seg in enumerate(segments):
        seg.export(str(SLICED_DIR / f"seg_{i:04d}.wav"), format="wav")
        total += len(seg) / 1000

    print(f"   ✅ {len(segments)} segments, {total:.0f}s total")
    return len(segments)


def filter_segments():
    print(f"\n🔍 Step 4: Quality filtering...")
    segments = sorted(SLICED_DIR.glob("seg_*.wav"))
    removed = 0

    for seg_path in segments:
        audio, sr = sf.read(str(seg_path))
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))

        if rms_db < MIN_RMS_DB or rms_db > MAX_RMS_DB:
            seg_path.unlink()
            removed += 1
            continue

        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        energy = np.array([
            np.sum(audio[i:i+frame_len] ** 2)
            for i in range(0, len(audio) - frame_len, hop_len)
        ])
        if np.sum(energy > np.mean(energy) * 0.1) / max(len(energy), 1) < MIN_SPEECH_RATIO:
            seg_path.unlink()
            removed += 1

    remaining = sorted(SLICED_DIR.glob("seg_*.wav"))
    for i, p in enumerate(remaining):
        new = SLICED_DIR / f"seg_{i:04d}.wav"
        if p != new:
            p.rename(new)

    print(f"   Removed {removed}, remaining: {len(remaining)}")
    return len(remaining)


def transcribe(language="en"):
    print(f"\n📝 Step 5: Transcribing ({language})...")
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    segments = sorted(SLICED_DIR.glob("seg_*.wav"))
    lines = []

    for i, seg in enumerate(segments):
        try:
            result, info = model.transcribe(str(seg), language=language, vad_filter=True)
            text = " ".join(s.text.strip() for s in result).strip()
            if text and len(text) > 5:
                lines.append(f"{seg.absolute()}|speaker1|{language}|{text}")
                if (i + 1) % 20 == 0:
                    print(f"   [{i+1}/{len(segments)}] {text[:60]}...")
        except Exception as e:
            print(f"   ⚠️ {seg.name}: {e}")

    DATASET_DIR.mkdir(exist_ok=True)
    list_path = DATASET_DIR / "transcription.list"
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"   ✅ {len(lines)} segments transcribed → {list_path}")
    return list_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--language", "-l", default="en")
    parser.add_argument("--skip-denoise", action="store_true")
    parser.add_argument("--silence-thresh", type=int, default=-38)
    args = parser.parse_args()

    print("🎙️  Voice Clone — Audio Processing")
    print("=" * 50)
    start = datetime.now()

    wav = find_wav_file(args.input)
    print(f"\n📂 Input: {wav} ({get_duration_min(wav):.1f} min)")

    cleaned = denoise(wav, skip=args.skip_denoise)
    normalized = normalize(cleaned)
    slice_audio(normalized, args.silence_thresh)
    filter_segments()
    transcribe(args.language)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n✅ Done in {elapsed:.1f} minutes")
    print(f"   Next: make train")


if __name__ == "__main__":
    main()
