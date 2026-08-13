![ComfyUI-QwenVL-Mod RunPod](https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/img/bannercu128.png)

# ComfyUI - QwenVL-Mod Custom Pod - CUDA 12.8

Custom ComfyUI based on `runpod/comfyui:cuda12.8`, enhanced with QwenVL-Mod and WAN 2.2 video generation. Compatible with RTX 4090 (Ada) and RTX 5090 (Blackwell).

**Template**: `OneClick ComfyUI WAN2.2 Qwen3VL CUDA 12.8`

---

## 🚀 Features

- **WAN 2.2 Video**: T2V, I2V, Storyboard, MMAudio workflows
- **Multilingual prompts** with visual style detection
- **GGUF backend** via llama-cpp-python CUDA 12.8
- **Sage Attention**, FP16 accumulation, async offload
- **TensorRT** upscaling and frame interpolation (CUDA 12 wheels pre-baked)
- **Persistent** `/workspace` (models survive restarts)
- **Auto-update** ComfyUI core at boot

---

## 📦 What's Included

### Base Image (runpod/comfyui:cuda12.8)
CUDA 12.8, PyTorch 2.10+cu128, Python 3.12, ComfyUI core, Manager (legacy UI), KJNodes, Civicomfy, RunpodDirect, FileBrowser, Jupyter, SSH.

### Custom Nodes (21)
QwenVL-Mod, RIFE-TensorRT-Auto, Upscaler-TensorRT-Auto, HuggingFace, GGUF, Euler-Smea-Dyn-Sampler, was-node-suite, VideoHelperSuite, rgthree-comfy, Easy-Use, Frame-Interpolation, mxToolkit, PainterI2V, PainterLongVideo, find-perfect-resolution, Selectors, MMAudio, VFI, WanMoeKSampler, comfy_mtb, comfy-tagcomplete.

### Pre-baked Models (~5GB)
- VAE: `wan_2.1_vae.safetensors`, `sdxl_vae.safetensors`
- Upscale: `2xLexicaRRDBNet.pth`, `2xLexicaRRDBNet_Sharp.pth`
- Text encoder: `nsfw_wan_umt5-xxl_fp8_scaled.safetensors`

### Downloaded at Boot (~30GB, persistent)
- `models/diffusion_models/wan22RemixT2VI2V_i2vHighV30.safetensors`
- `models/diffusion_models/wan22RemixT2VI2V_i2vLowV30.safetensors`

> Background download on first boot. ComfyUI starts immediately; models appear when ready. No re-download on restart.

### Workflows (13)
WAN 2.2 T2V/I2V/SVI (GGUF variants), Full MMAudio, AutoPrompt Story, PMP LoRaStack.

---

## 🛠️ Requirements

- **GPU**: RTX 4090 (Ada) / RTX 5090 (Blackwell) or any CUDA 12.8 card
- **VRAM**: 24GB+ (4090) / 32GB+ (5090) recommended
- **Storage**: 100GB+ SSD

---

## 🚀 Quick Start

1. **Deploy**: Select "OneClick ComfyUI WAN2.2 Qwen3VL CUDA 12.8"
2. **First boot**: ComfyUI copies to `/workspace` (~30s), WAN models download in background
3. **Access**:
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

## � Persistence

`/workspace/runpod-slim/ComfyUI` survives restarts: models, nodes, workflows, output, user data. No re-downloads on restart.

---

## 📊 CUDA 12.8 vs 13.0

| | CUDA 12.8 | CUDA 13.0 |
|---|---|---|
| **GPU** | RTX 4090 + 5090 | RTX 5090+ |
| **TensorRT** | 10.13.3.9 | 10.15.1.29 |
| **llama-cpp-python** | v0.3.45+cu128 | v0.3.45+cu130 |
| **Base image** | `runpod/comfyui:cuda12.8` | `runpod/comfyui:cuda13.0` |

Choose CUDA 12.8 if you need RTX 4090 support. Choose CUDA 13.0 for RTX 5090-only with latest TensorRT.

---

Based on RunPod official template with QwenVL-Mod enhancements.
