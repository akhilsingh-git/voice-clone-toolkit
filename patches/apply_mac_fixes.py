#!/usr/bin/env python3
"""
Auto-apply all Mac/Apple Silicon patches to GPT-SoVITS.
Fixes every float16 issue across training, inference, and API.

Run: python patches/apply_mac_fixes.py
"""

import os, sys, re
from pathlib import Path


def patch_file(filepath, replacements):
    path = Path(filepath)
    if not path.exists():
        print(f"  ⚠️  Skip (not found): {filepath}")
        return False
    content = path.read_text()
    patched = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            patched = True
    if patched:
        path.write_text(content)
        print(f"  ✅ {path.name}")
    else:
        print(f"  ⏭️  Already patched: {path.name}")
    return patched


def main():
    gpt = Path("GPT-SoVITS")
    if not gpt.exists():
        print("❌ GPT-SoVITS not found. Run 'make gpt-sovits' first.")
        sys.exit(1)

    print("🩹 Applying Mac/Apple Silicon patches...\n")

    # ============================================================
    # Patch 1: Training — HuBERT feature extraction (2-get-hubert-wav32k.py)
    # ============================================================
    print("Training scripts:")
    hubert = gpt / "GPT_SoVITS" / "prepare_datasets" / "2-get-hubert-wav32k.py"
    patch_file(str(hubert), [
        ("model = model.half().to(device)", "model = model.float().to(device)"),
        ("model = model.to(device)", "model = model.float().to(device)"),
        ("tensor_wav16 = tensor_wav16.to(device)", "tensor_wav16 = tensor_wav16.float().to(device)"),
    ])

    # Patch 2: Training — Speaker verification (2-get-sv.py)
    sv = gpt / "GPT_SoVITS" / "prepare_datasets" / "2-get-sv.py"
    patch_file(str(sv), [
        ("model = model.half().to(device)", "model = model.float().to(device)"),
    ])

    # ============================================================
    # Patch 3: Inference WebUI (inference_webui.py)
    # ============================================================
    print("\nInference WebUI:")
    inf = gpt / "GPT_SoVITS" / "inference_webui.py"
    if inf.exists():
        content = inf.read_text()
        target = 'ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)'
        if target in content and "ssl_model.model = ssl_model.model.float()" not in content:
            content = content.replace(
                f"            {target}",
                f"            ssl_model.model = ssl_model.model.float()\n"
                f"            wav16k = wav16k.float()\n"
                f"            {target}"
            )
            # Also fix the else branch for wav16k
            content = content.replace(
                "                wav16k = wav16k.to(device)",
                "                wav16k = wav16k.float().to(device)"
            )
            inf.write_text(content)
            print(f"  ✅ inference_webui.py")
        else:
            print(f"  ⏭️  Already patched: inference_webui.py")

    # ============================================================
    # Patch 4: API / TTS inference pack (TTS.py) — THE BIG ONE
    # Forces float32 on ALL model loads regardless of is_half config
    # ============================================================
    print("\nTTS API (TTS_infer_pack/TTS.py):")
    tts = gpt / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
    if tts.exists():
        content = tts.read_text()
        changes = 0

        # Force is_half = False everywhere
        old = 'self.is_half = self.configs.get("is_half", False)'
        new = 'self.is_half = False  # Force float32 on Mac'
        if old in content:
            content = content.replace(old, new)
            changes += 1

        # Replace all .half() with .float()
        if '.half()' in content:
            content = content.replace('.half()', '.float()')
            changes += 1

        # Replace float16 dtype refs
        content = content.replace('torch.float16', 'torch.float32')
        content = content.replace('np.float16', 'np.float32')

        # Force float32 on model-to-device calls
        for model_var in ['self.cnhuhbert_model', 'self.bert_model', 'vits_model', 't2s_model']:
            old_line = f'{model_var} = {model_var}.to(self.configs.device)'
            new_line = f'{model_var} = {model_var}.float().to(self.configs.device)'
            if old_line in content and new_line not in content:
                content = content.replace(old_line, new_line)
                changes += 1

        if changes > 0:
            tts.write_text(content)
            print(f"  ✅ TTS.py ({changes} fixes)")
        else:
            print(f"  ⏭️  Already patched: TTS.py")

    # ============================================================
    # Patch 5: NLTK data
    # ============================================================
    print("\nNLTK data:")
    try:
        import nltk
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('cmudict', quiet=True)
        print("  ✅ NLTK data installed")
    except Exception as e:
        print(f"  ⚠️  Run manually: python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")

    print("\n✅ All patches applied!")


if __name__ == "__main__":
    main()
