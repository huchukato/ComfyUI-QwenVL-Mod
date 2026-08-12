![ComfyUI-QwenVL-Mod LTX 2.3](https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/img/bannerltx23.png)

# OneClick - ComfyUI - LTX 2.3 Uncensored - Qwen3VL

Custom ComfyUI based on `runpod/comfyui:cuda13.0`, enhanced with QwenVL-Mod and native LTX 2.3 video+audio generation with Qwen3-VL auto-prompting. Uncensored 10Eros setup for NSFW I2V.

**Template**: `OneClick - ComfyUI - LTX 2.3 Uncensored - Qwen3VL`

**Docker image**: `huchukato/comfyui-qwenvl-runpod:cu13-ltx`

---

## 🚀 Features

- **LTX 2.3 native video+audio**: I2V with synchronized audio generation
- **10Eros v1 uncensored**: merge of Sulphur 2 optimized for NSFW I2V
- **Gemma 3 12B abliterated**: uncensored text encoder LoRA
- **Multilingual prompts** with visual style detection via QwenVL-Mod
- **Direct QwenVL → CLIPTextEncode**: LTX prompt enhancer bypassed (preserves prompt intent)
- **GGUF backend** via llama-cpp-python CUDA 13
- **Sage Attention**, FP16 accumulation, async offload
- **TensorRT** upscaling and frame interpolation
- **Spatial + temporal upscalers** included
- **Persistent** `/workspace` (models survive restarts)
- **ComfyUI v0.31.0+** forced at boot
- **hf-transfer** for fast multi-connection downloads

---

## 📦 What's Included

### Base Image (`runpod/comfyui:cuda13.0`)
CUDA 13.0, PyTorch 2.10+cu130, Python 3.12, ComfyUI core, Manager, KJNodes, Civicomfy, RunpodDirect, FileBrowser, Jupyter, SSH.

### Custom Nodes
QwenVL-Mod, ComfyUI-LTXVideo (Lightricks), ComfyUI-RIFE-TensorRT-Auto, ComfyUI-Upscaler-TensorRT-Auto, ComfyUI-HuggingFace, comfy-tagcomplete, Euler-Smea-Dyn-Sampler, was-node-suite, ComfyUI-VideoHelperSuite, rgthree-comfy, ComfyUI-Easy-Use, ComfyUI-Frame-Interpolation, comfyui-find-perfect-resolution, ComfyUI-Crystools-MonitorOnly.

> LTX 2.3 core nodes are built into ComfyUI 0.30.0+. `ComfyUI-LTXVideo` adds workflow-specific nodes.

### Workflow (1)
- `LTX23-I2VA-Qwen3VL.json` — I2V+Audio with QwenVL auto-prompt, 10Eros, dual-stage with spatial upscaler

### Models auto-downloaded at first boot (~40 GB, persistent)

| Subfolder | Model | Source | Size |
|---|---|---|---|
| `checkpoints` | `10Eros_v1-fp8mixed_learned.safetensors` | TenStrip/LTX2.3-10Eros (uncensored) | ~29 GB |
| `text_encoders` | `gemma_3_12B_it_fp4_mixed.safetensors` | Comfy-Org/ltx-2 | ~9.4 GB |
| `loras` | `gemma-3-12b-it-abliterated_lora_rank64_bf16` | Comfy-Org/ltx-2 (uncensors encoder) | ~628 MB |
| `loras/ltx23` | `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe` | TenStrip (distilled, cond_safe for 10Eros) | ~662 MB |
| `upscale_models` | `ltx-2.3-spatial-upscaler-x2-1.1` | Lightricks/LTX-2.3 (2x resolution) | ~996 MB |
| `upscale_models` | `ltx-2.3-temporal-upscaler-x2-1.0` | Lightricks/LTX-2.3 (2x frames) | ~262 MB |

> ComfyUI starts immediately; models download in background. No re-download on restart.

---

## 🛠️ Requirements

- **GPU**: RTX 5090+ or any CUDA 13.0 card
- **VRAM**: 24 GB+ recommended
- **Storage**: 100 GB+ SSD

---

## 🔑 Hugging Face Token (optional but recommended)

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Quick Start

1. **Deploy**: Select `OneClick - ComfyUI - LTX 2.3 Uncensored - Qwen3VL`
2. **First boot**: ComfyUI copies to `/workspace`, models download in background
3. **Load workflow**: `ComfyUI > Load > LTX23-I2VA-Qwen3VL.json`
4. **Access**: ComfyUI `:8188` · JupyterLab `:8888` · FileBrowser `:8080` · SSH `ssh root@pod-ip`

---

## ⚙️ ComfyUI Args

```
--disable-auto-launch
--fast fp16_accumulation
--use-sage-attention
--reserve-vram 2
--cuda-malloc
--async-offload
```

---

## 🎬 Prompting Notes

- Use QwenVL-Mod **🎥 LTX 2.3 NSFW I2V** preset (mandatory audio instructions)
- QwenVL output goes **directly** to CLIPTextEncode — LTX enhancer bypassed
- LTX 2.3 generates **synchronized audio**: always specify dialogue (in quotes), tone of voice, and ambient sounds
- Frame count must be N×8+1 (121 = ~5s, 201 = ~8.4s). Use temporal upscaler for longer videos
- Native resolution ~768px short edge. Use spatial upscaler for higher resolution
- Do NOT use `distilled-lora-384` with 10Eros — use the `condsafe` variant only

---

## 🔄 Persistence

`/workspace/runpod-slim/ComfyUI` survives restarts: models, nodes, workflows, outputs. No re-downloads on restart.

---

Based on the RunPod official template with QwenVL-Mod and LTX 2.3 enhancements.
