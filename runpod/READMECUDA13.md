![ComfyUI-QwenVL-Mod RunPod](https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/img/bannercu13.png)

# ComfyUI - QwenVL-Mod Custom Pod - CUDA 13.0

Custom ComfyUI based on `runpod/comfyui:cuda13.0`, enhanced with QwenVL-Mod and WAN 2.2 video generation.

**Template**: `OneClick ComfyUI WAN 2.2 Qwen3VL CUDA 13.0`

---

## 🚀 Features

- **WAN 2.2 Video**: T2V, I2V, Storyboard, MMAudio workflows
- **Multilingual prompts** with visual style detection
- **GGUF backend** via llama-cpp-python CUDA 13
- **Sage Attention**, FP16 accumulation, async offload
- **TensorRT** upscaling and frame interpolation
- **Persistent** `/workspace` (models survive restarts)

---

## 📦 What's Included

### Base Image (runpod/comfyui:cuda13.0)
CUDA 13.0, PyTorch 2.10+cu130, Python 3.12, ComfyUI core, Manager (legacy UI), KJNodes, Civicomfy, RunpodDirect, FileBrowser, Jupyter, SSH.

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

- **GPU**: RTX 5090 / 4090 or any CUDA 13.0 card
- **VRAM**: 32GB+ recommended
- **Storage**: 100GB+ SSD

---

## 🚀 Quick Start

1. **Deploy**: Select "Custom-ComfyUI-WAN2.2-Qwen3VL-CUDA13"
2. **First boot**: ComfyUI copies to `/workspace` (~30s), WAN models download in background
3. **Access**:
   - ComfyUI: `http://pod-ip:8188`
   - JupyterLab: `http://pod-ip:8888`
   - FileBrowser: `http://pod-ip:8080`
   - SSH: `ssh root@pod-ip`

---

## � ComfyUI Args

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

## 🔄 Persistence

`/workspace/runpod-slim/ComfyUI` survives restarts: models, nodes, workflows, output, user data. No re-downloads on restart.

---

Based on RunPod official template with QwenVL-Mod enhancements.
