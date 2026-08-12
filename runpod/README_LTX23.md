![ComfyUI-QwenVL-Mod LTX 2.3](https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/img/bannerltx23.png)

# OneClick - ComfyUI - LTX 2.3 Uncensored - Qwen3VL

Custom ComfyUI based on `runpod/comfyui:cuda13.0`, enhanced with QwenVL-Mod and native LTX 2.3 video+audio generation with Qwen3-VL auto-prompting. Uncensored 10Eros setup for NSFW I2V.

**Template**: `OneClick - ComfyUI - LTX 2.3 Uncensored - Qwen3VL`

**Docker image**: `huchukato/comfyui-qwenvl-runpod:cu13-ltx`

---

## 🚀 Features

- **LTX 2.3 native video+audio**: I2V workflow with synchronized audio generation
- **10Eros v1 uncensored**: merge of Sulphur 2 optimized for NSFW I2V
- **Gemma 3 12B abliterated**: uncensored text encoder LoRA
- **Multilingual prompts** with visual style detection via QwenVL-Mod
- **Direct QwenVL → CLIPTextEncode**: no LTX prompt enhancer in between (preserves original prompt intent)
- **GGUF backend** via llama-cpp-python CUDA 13
- **Sage Attention**, FP16 accumulation, async offload
- **TensorRT** upscaling and frame interpolation
- **Spatial + temporal upscalers** for resolution and frame count doubling
- **Persistent** `/workspace` (models survive restarts)
- **ComfyUI v0.31.0+** forced at boot
- **hf-transfer** for fast multi-connection Hugging Face downloads

---

## 📦 What's Included

### Base Image (`runpod/comfyui:cuda13.0`)
CUDA 13.0, PyTorch 2.10+cu130, Python 3.12, ComfyUI core, Manager (legacy UI), KJNodes, Civicomfy, RunpodDirect, FileBrowser, Jupyter, SSH.

### Custom Nodes
QwenVL-Mod, ComfyUI-LTXVideo (Lightricks), ComfyUI-RIFE-TensorRT-Auto, ComfyUI-Upscaler-TensorRT-Auto, ComfyUI-HuggingFace, comfy-tagcomplete, Euler-Smea-Dyn-Sampler, was-node-suite, ComfyUI-VideoHelperSuite, rgthree-comfy, ComfyUI-Easy-Use, ComfyUI-Frame-Interpolation, comfyui-find-perfect-resolution, ComfyUI-Crystools-MonitorOnly.

> LTX 2.3 core nodes (LTXVConditioning, LTXVImgToVideoInplace, LTXVAudioVAEDecode, etc.) are built into ComfyUI 0.30.0+. The `ComfyUI-LTXVideo` node from Lightricks adds workflow-specific nodes.

### Workflow (1)
Downloaded automatically at boot from the latest repository version:

- `LTX23-I2VA-Qwen3VL.json` — Image-to-Video+Audio with QwenVL auto-prompt, 10Eros checkpoint, dual-stage generation with spatial upscaler

### Models auto-downloaded at first boot (~40 GB, persistent)

| Subfolder | Model | Source | Size |
|---|---|---|---|
| `models/checkpoints` | `10Eros_v1-fp8mixed_learned.safetensors` | TenStrip/LTX2.3-10Eros (uncensored) | ~29 GB |
| `models/text_encoders` | `gemma_3_12B_it_fp4_mixed.safetensors` | Comfy-Org/ltx-2 | ~9.4 GB |
| `models/loras` | `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | Comfy-Org/ltx-2 (uncensors text encoder) | ~628 MB |
| `models/loras/ltx23` | `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` | TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments | ~662 MB |
| `models/upscale_models` | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Lightricks/LTX-2.3 (doubles resolution) | ~996 MB |
| `models/upscale_models` | `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | Lightricks/LTX-2.3 (doubles frame count) | ~262 MB |

> First boot downloads ~40 GB in the background. ComfyUI starts immediately; models appear in the Loaders when ready. No re-download on restart.

---

## 🛠️ Requirements

- **GPU**: RTX 5090+ or any CUDA 13.0 card
- **VRAM**: 24 GB+ recommended (10Eros FP8 + Gemma FP4)
- **Storage**: 100 GB+ SSD

---

## 🔑 Hugging Face Token (optional but recommended)

Public downloads work without a token, but setting a `HF_TOKEN` helps avoid rate limits.

In the RunPod template environment variables add:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The boot script detects `HF_TOKEN` automatically and passes it to every Hugging Face download.

---

## 🚀 Quick Start

1. **Deploy**: Select `OneClick - ComfyUI - LTX 2.3 Uncensored - Qwen3VL`
2. **First boot**: ComfyUI copies to `/workspace`, then LTX 2.3 models download in background
3. **Load a workflow** from `ComfyUI > Load > LTX23-I2VA-Qwen3VL.json`
4. **Access**:
   - ComfyUI: `http://pod-ip:8188`
   - JupyterLab: `http://pod-ip:8888`
   - FileBrowser: `http://pod-ip:8080`
   - SSH: `ssh root@pod-ip`

---

## ⚙️ ComfyUI Args

Pre-populated in `/workspace/runpod-slim/comfyui_args.txt`:

```
--disable-auto-launch
--fast fp16_accumulation
--use-sage-attention
--reserve-vram 2
--cuda-malloc
--async-offload
```

Edit via FileBrowser or Jupyter to customize.

---

## 🎬 LTX 2.3 Prompting Notes

- Use the QwenVL-Mod **🎥 LTX 2.3 NSFW I2V** preset for auto-prompting with mandatory audio instructions
- The QwenVL output goes **directly** to CLIPTextEncode — the LTX `TextGenerateLTX2Prompt` enhancer is bypassed to preserve the original prompt intent
- LTX 2.3 generates **synchronized audio** alongside video. The preset always includes:
  - Dialogue in direct quotes (when speech is present)
  - Explicit "no speech" + ambient sounds (when no dialogue)
  - Tone of voice and ambient sound descriptions
  - NSFW-appropriate audio (moans, gasps, wet sounds, etc.)

### LoRA Setup

| LoRA | Applies to | Strength | Purpose |
|---|---|---|---|
| `gemma-3-12b-it-abliterated_lora_rank64_bf16` | Text encoder (Gemma 3) | 1.0 | Uncensors text encoder for NSFW |
| `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe` | Model (diffusion) | 0.5 | Distilled acceleration, cond_safe variant for 10Eros |

> Do NOT use the standard `ltx-2.3-22b-distilled-lora-384` with 10Eros — it damages the finetune. Always use the `condsafe` variant.

### Frame Length Guide

LTX 2.3 uses 8:1 temporal compression. Frame count must be N×8+1:

| Frames | Duration (24fps) | VRAM |
|---|---|---|
| 41 | ~1.7s | Low |
| 81 | ~3.4s | Medium |
| 121 | ~5s | Medium |
| 161 | ~6.7s | High |
| 201 | ~8.4s | High |
| 241 | ~10s | Very High |

For longer videos: generate 121-201 frames, then apply the **temporal upscaler** to double the frame count.

### Resolution Guidance

LTX 2.3 native resolution uses ~768px short edge. For higher resolution, apply the **spatial upscaler** after generation (already wired in the workflow).

Examples:
- 768x1344 (portrait)
- 896x1152
- 960x1280
- 1024x1024

Avoid direct 1080p. For best quality, generate at native resolution and then upscale / interpolate frames with the bundled TensorRT nodes.

---

## 🔄 Persistence

`/workspace/runpod-slim/ComfyUI` survives restarts: models, nodes, workflows, outputs, and user data. No re-downloads on restart.

---

## 🏗️ Build

```bash
cd runpod
./build-and-push-CU13-LTX.sh
```

Builds and pushes `huchukato/comfyui-qwenvl-runpod:cu13-ltx` to Docker Hub.

---

Based on the RunPod official template with QwenVL-Mod and LTX 2.3 enhancements.
