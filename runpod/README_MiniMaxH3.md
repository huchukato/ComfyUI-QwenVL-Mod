![ComfyUI-QwenVL-Mod MiniMax H3](https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/img/bannerminimax.png)

# OneClick - ComfyUI - MiniMax H3 Turbo Uncensored

Custom ComfyUI based on `runpod/comfyui:cuda13.0`, enhanced with QwenVL-Mod and native MiniMax H3 video+audio generation with Qwen3-VL auto-prompting.

**Template**: `OneClick - ComfyUI - MiniMax H3 Turbo Uncensored`

---

## 🚀 Features

- **MiniMax H3 native video+audio**: T2VA, I2VA, FL2VA / R2VA workflows
- **Built-in audio generation**: no separate MMAudio node required
- **Multilingual prompts** with visual style detection via QwenVL-Mod
- **GGUF backend** via llama-cpp-python CUDA 13
- **Sage Attention**, FP16 accumulation, async offload
- **TensorRT** upscaling and frame interpolation
- **Persistent** `/workspace` (models survive restarts)
- **ComfyUI v0.32.0+** forced at boot (MiniMax H3 requirement)

---

## 📦 What's Included

### Base Image (`runpod/comfyui:cuda13.0`)
CUDA 13.0, PyTorch 2.10+cu130, Python 3.12, ComfyUI core, Manager, KJNodes, Civicomfy, RunpodDirect, FileBrowser, Jupyter, SSH.

### Custom Nodes
QwenVL-Mod, Larryvrh/ComfyUI-MiniMax-H3-Turbo, ComfyUI-RIFE-TensorRT-Auto, ComfyUI-Upscaler-TensorRT-Auto, ComfyUI-HuggingFace, was-node-suite, ComfyUI-VideoHelperSuite, rgthree-comfy, ComfyUI-Easy-Use, ComfyUI-Frame-Interpolation, comfyui-find-perfect-resolution.

> MiniMax H3 support is built into ComfyUI 0.30.0. The Turbo LoRA requires the `ComfyUI-MiniMax-H3-Turbo` custom node.

### Workflows (8)
Downloaded automatically at boot:

- `MiniMaxH3-I2VA-Qwen3VL.json`
- `MiniMaxH3-T2VA-Qwen3VL.json`
- `MiniMaxH3-FL2VA-Qwen3VL.json`
- `MiniMaxH3-R2VA-Qwen3VL.json`
- `MiniMaxH3-Turbo-I2VA-Qwen3VL.json`
- `MiniMaxH3-Turbo-T2VA-Qwen3VL.json`
- `MiniMaxH3-Turbo-FL2VA-Qwen3VL.json`
- `MiniMaxH3-Turbo-R2VA-Qwen3VL.json`

### Models auto-downloaded at first boot (~70 GB, persistent)

| Subfolder | Model | Source | Size |
|---|---|---|---|
| `vae` | `minimax_h3_video_vae_fp16` | Comfy-Org/MiniMax-H3 | ~5 GB |
| `vae` | `minimax_h3_audio_vae_fp32` | Comfy-Org/MiniMax-H3 | ~0.6 GB |
| `diffusion_models` | `minimax_h3_fl2va_pruned_int8_convrot` | Comfy-Org/MiniMax-H3 | ~20 GB |
| `diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot` | Comfy-Org/MiniMax-H3 (R2VA) | ~20 GB |
| `text_encoders` | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot` | ethanfel (uncensored) | ~27 GB |
| `loras` | `minimax_h3_turbo_v4_step600_ema` | larryvrh/MiniMax-H3-Turbo-Lora | ~0.7 GB |

> ComfyUI starts immediately; models download in background. No re-download on restart.

---

## 🛠️ Requirements

- **GPU**: RTX 5090+ or any CUDA 13.0 card
- **VRAM**: 32 GB+ recommended
- **Storage**: 150 GB+ SSD

---

## 🔑 Hugging Face Token (optional but recommended)

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Quick Start

1. **Deploy**: Select `OneClick - ComfyUI - MiniMax H3 Turbo Uncensored`
2. **First boot**: ComfyUI copies to `/workspace`, models download in background
3. **Load workflow**: `ComfyUI > Load > MiniMaxH3-*-Qwen3VL.json`
4. **Access**: ComfyUI `:8188` · JupyterLab `:8888` · FileBrowser `:8080` · SSH `ssh root@pod-ip`

---

## ⚙️ ComfyUI Args

```
--disable-auto-launch --fast fp16_accumulation --use-sage-attention --cuda-malloc --async-offload
```

---

## 🎬 Prompting Notes

- Use QwenVL-Mod **MiniMax H3 NSFW (5s/10s/15s)** presets for native video+audio prompts
- Presets produce the official three-field format: `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`
- Audio is generated natively; describe sounds explicitly in the prompt
- Native resolution: **768px short edge**, long edge capped at **1344px**, multiples of 32
- Avoid direct 1080p. Generate at native resolution, then upscale/interpolate with TensorRT nodes

---

## 🔄 Persistence

`/workspace/runpod-slim/ComfyUI` survives restarts: models, nodes, workflows, outputs. No re-downloads on restart.

---

Based on the RunPod official template with QwenVL-Mod and MiniMax H3 enhancements.
