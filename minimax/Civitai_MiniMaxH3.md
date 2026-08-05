# [MiniMax H3] NSFW T2VA/I2VA/FLF/R2VA Workflows 🎬 Auto Prompt | Native Audio | TensorRT Upscale | RIFE Interpolation

![MiniMax H3 Qwen3VL](https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/img/bannerminimax.png)

ComfyUI-QwenVL-Mod — Enhanced Vision-Language with MiniMax H3
Version 2.4 (2026/08/04) — 🎬 MiniMax H3 Native Video+Audio + Qwen3-VL Auto-Prompting

---

## 🌟 What is ComfyUI-QwenVL-Mod?

A powerful enhanced vision-language node for ComfyUI that combines **Qwen3-VL** models with **MiniMax H3** video generation workflows. Features multilingual support, visual style detection, native stereo audio, and NSFW capabilities for professional AI content creation.

Think: *"Your all-in-one solution for intelligent prompt enhancement and video+audio generation with MiniMax H3!"*

---

## 🎬 Key Features

### 🚀 MiniMax H3 Video+Audio Generation

- **T2VA** (Text-to-Video+Audio): Generate video with native stereo audio from text
- **I2VA** (Image-to-Video+Audio): Animate a first-frame image with audio
- **FLF** (First-Last-Frame): Generate the transition between two keyframes
- **R2VA** (Reference-to-Video): Lock character identity, style, motion, or voice using reference images/videos/audio

### 🧠 Qwen3-VL Auto-Prompting

- **Multilingual**: Write your prompt in **any language** — Qwen3-VL translates and converts it
- **Auto-format**: Generates the official MiniMax H3 three-field format (`integrated_multimodal_description` + `overall_soundscape` + `non_diegetic_music`)
- **Visual style detection**: 12+ artistic styles (photorealistic, cinematic, anime, 3D CG, claymation, vintage film, watercolor, fantasy, etc.)
- **Smart caching**: Performance optimization with Fixed Seed Mode
- **GGUF backend**: Efficient local model inference with quantization support

### 🔊 Native Stereo Audio

- **No separate audio node needed** — MiniMax H3 generates video and audio jointly in a single forward pass
- Voice, sound effects, and music modeled together, not layered on afterward
- Describe sounds in your prompt and the model generates them natively

### 🎨 NSFW Support

- Comprehensive content generation without restrictions
- Dedicated NSFW presets (5s / 10s / 15s) with explicit diegetic soundscape
- Natural progression, style adaptation, consistent characters

---

## 📦 What's Included — 4 Workflows

| # | Workflow | Mode | Inputs | Description |
|---|---|---|---|---|
| 1 | `MiniMaxH3-T2VA-Qwen3VL.json` | 📝 **T2VA** | text only | Text-to-video+audio. Simplest workflow. |
| 2 | `MiniMaxH3-I2VA-Qwen3VL.json` | 🖼️ **I2VA / FL2VA / L2VA** | text + first-frame image (optional last-frame) | Image-to-video. Covers first-frame, last-frame, and first+last-frame. |
| 3 | `MiniMaxH3-I2VA-FLF-Qwen3VL.json` | 🎞️ **FLF** | text + first-frame + last-frame | First-Last-Frame to video. Includes TensorRT upscale + RIFE frame interpolation for 48 fps output. |
| 4 | `MiniMaxH3-R2VA-Qwen3VL.json` | 🎬 **R2VA / Reference** | text + reference image/video/audio | Reference-to-video. Lock identity, style, motion, camera, or voice using up to 9 ref images, 3 ref videos, 3 ref audio clips. |

> Workflows 2 and 3 include **TensorRT upscaling** (RealESRGAN x4) and **RIFE frame interpolation** (rife49) for 48 fps high-resolution output.

---

## 🎯 QwenVL-Mod NSFW Presets

The workflows include built-in NSFW presets for the Qwen3-VL prompt enhancer:

| Preset | Duration | Use case |
|---|---|---|
| `🎬 MiniMax H3 NSFW (5s)` | 5 seconds | Short clips, fast iteration |
| `🎬 MiniMax H3 NSFW (10s)` | 10 seconds | Standard duration |
| `🎬 MiniMax H3 NSFW (15s)` | 15 seconds | Maximum duration |

### What the presets produce

- `[Shot 1]` with style + initial composition (per official MiniMax H3 guide)
- Official camera motion vocabulary (Zoom In/Out, Push In/Pull Out, Pan, Tilt, Dolly, Crane, Orbit, etc.)
- Speaker IDs/tags for dialogues
- Explicit diegetic soundscape (breaths, moans, skin contact, ambient sounds)
- Optional non-diegetic music (defaults to None — rely only on the diegetic soundscape)

> SFW presets are also available. Edit the preset dropdown in the QwenVL node to switch.

---

## 🎮 Usage Examples

### Basic Text-to-Video (T2VA)
1. Load `MiniMaxH3-T2VA-Qwen3VL.json`
2. Write your prompt in any language
3. Select preset `🎬 MiniMax H3 NSFW (5s/10s/15s)`
4. Generate video with native audio

### Image-to-Video (I2VA)
1. Load `MiniMaxH3-I2VA-Qwen3VL.json`
2. Upload your first-frame image
3. Write what happens next (in any language)
4. Generate animated video with audio

### First-Last-Frame (FLF)
1. Load `MiniMaxH3-I2VA-FLF-Qwen3VL.json`
2. Upload first-frame and last-frame images
3. Describe the transition between them
4. Generate the interpolated video at 48 fps with TensorRT upscale + RIFE

### Reference-to-Video (R2VA)
1. Load `MiniMaxH3-R2VA-Qwen3VL.json`
2. Upload reference images/videos/audio (up to 9 images, 3 videos, 3 audio clips)
3. Reference them by tag in your prompt: `<Picture 1>`, `<Video 1>`, `<Audio 1>`
4. Generate video with locked identity/style/voice

---

## 🔧 Technical Specifications

### ⚡ Performance

- **Output**: 768p, 24 fps (native), up to ~15 seconds
- **Audio**: Native stereo, generated jointly with video
- **Upscale**: TensorRT RealESRGAN x4 (I2VA + FLF workflows)
- **Frame interpolation**: RIFE rife49 → 48 fps (I2VA + FLF workflows)
- **Sage Attention**: FP16 accumulation, async offload
- **Smart caching**: Reuse prompts with same inputs, Fixed Seed Mode for text-only caching

### 🎨 Model Support

- **Qwen3-VL 4B**: 7 GGUF variants (2.38 GB – 4.28 GB)
- **Qwen3-VL 8B**: 7 GGUF variants (4.8 GB – 8.71 GB)
- **Qwen3.5**: 4B / 9B / 27B (uncensored, heretic, unsloth)
- **HF Models**: Josiefed, official, Heretic-Stable variants
- **Quantization**: Q4_K_S, Q5_K_S, FP16, INT8

### 🌐 Multilingual Capabilities

- **Input languages**: Any language supported
- **Auto-translation**: Automatic translation to optimized English
- **Style detection**: Works with multilingual prompts
- **Cultural adaptation**: Context-aware prompt enhancement

---

## 📦 Installation

### Requirements

- **ComfyUI**: v0.30.0+ (required for MiniMax H3 native support)
- **GPU**: RTX 5090+ or any CUDA 12.8/13.0 card
- **VRAM**: 32 GB+ recommended (int8 models)
- **Storage**: 100 GB+ SSD for models
- **Python**: 3.10+

### Quick Install

1. Download: [ComfyUI-QwenVL-Mod](https://github.com/huchukato/ComfyUI-QwenVL-Mod) (latest version)
2. Extract to `ComfyUI/custom_nodes/ComfyUI-QwenVL-Mod`
3. Install requirements: `pip install -r requirements.txt`
4. Restart ComfyUI
5. Load included workflows

### Custom Nodes Required

| Custom Node | Used by | Repo |
|---|---|---|
| **ComfyUI-QwenVL-Mod** | All workflows (Qwen3-VL prompt enhancer) | [huchukato/ComfyUI-QwenVL-Mod](https://github.com/huchukato/ComfyUI-QwenVL-Mod) |
| **ComfyUI-RIFE-TensorRT-Auto** | I2VA, FLF (frame interpolation) | [huchukato/ComfyUI-RIFE-TensorRT-Auto](https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto) |
| **ComfyUI-Upscaler-TensorRT-Auto** | I2VA, FLF (upscaling) | [huchukato/ComfyUI-Upscaler-TensorRT-Auto](https://github.com/huchukato/ComfyUI-Upscaler-TensorRT-Auto) |
| **ComfyUI-VideoHelperSuite** | I2VA, FLF (VHS_VideoCombine) | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |
| **ComfyUI-Easy-Use** | I2VA, FLF (easy showAnything) | [yolain/ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) |
| **comfyui-find-perfect-resolution** | All workflows (ResolutionSelector) | [ashtar1984/comfyui-find-perfect-resolution](https://github.com/ashtar1984/comfyui-find-perfect-resolution) |
| **was-node-suite-comfyui** | R2VA (ComfyMathExpression) | [ltdrdata/was-node-suite-comfyui](https://github.com/ltdrdata/was-node-suite-comfyui) |

### Models Required

All MiniMax H3 models from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) on Hugging Face.

**T2VA / I2VA / FLF (fl2va)**

| Subfolder | Model | Size |
|---|---|---|
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` | ~5 GB |
| `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` | ~0.6 GB |
| `models/diffusion_models/` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | ~21 GB |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | ~27 GB |

**R2VA (ref2va)** — same as above, except:

| Subfolder | Model | Size |
|---|---|---|
| `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | ~21 GB |

**Qwen3-VL Prompt Enhancer**

| Subfolder | Model |
|---|---|
| `models/LLM/` | `Qwen3-VL-8B-Heretic-Stable` (GGUF or HF) |

**TensorRT Engines (I2VA + FLF only)**

| Subfolder | Model |
|---|---|
| `models/upscale_models/` | `RealESRGAN_x4` (TensorRT engine) |
| `models/rife/` | `rife49_ensemble_True_scale_1_sim` (TensorRT engine) |

> TensorRT engines must be built for your specific GPU. See [ComfyUI-RIFE-TensorRT-Auto](https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto) and [ComfyUI-Upscaler-TensorRT-Auto](https://github.com/huchukato/ComfyUI-Upscaler-TensorRT-Auto) for build instructions.

### Download Links

- **VAE**: [video_vae_fp16](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) · [audio_vae_fp32](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors)
- **Diffusion (fl2va)**: [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors)
- **Diffusion (ref2va)**: [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
- **Text encoder**: [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors)

---

## 🎬 MiniMax H3 Prompting Notes

### How to Write Your Prompt

Describe the scene naturally. Be clear about the concepts below — Qwen3-VL handles the rest:

- **🎨 Visual style** (put it first): `photorealistic`, `cinematic`, `anime`, `3D CG`, `claymation`, `vintage film`, `watercolor`, `fantasy`
- **👥 Subjects**: number, gender, appearance, clothing, position, expression
- **🏃 Action / motion**: what happens, speed, interaction
- **🎥 Camera**: dolly, pan, zoom, static, handheld, crane, orbit
- **🌍 Environment**: setting, lighting, atmosphere, time of day
- **🔊 Audio** (important!): dialogue, breaths, moans, skin contact, ambient sounds, music

### Resolution Guidance

MiniMax H3 native canvas: **768 px short edge**, long edge capped at **1344 px**, multiples of **32**.

| megapixels | Aspect | Output |
|---|---|---|
| 0.5 | 16:9 | 960 x 544 |
| 0.8 | 16:9 | 1216 x 672 |
| 0.98 | 16:9 | 1344 x 768 |
| 1.0 | 16:9 | 1376 x 768 |

> Avoid direct 1080p. Generate at native resolution, then upscale with TensorRT nodes (I2VA + FLF workflows).

### Duration

Choose a preset: **5s / 10s / 15s**. The Math Expression node snaps the frame count to the model's 17-frame-per-block grid (17k+5 at 24 fps).

---

## 🐳 Docker / Cloud Ready

### OneClick RunPod Template

Prefer a ready-to-go environment? Use the **OneClick ComfyUI MiniMax H3 Qwen3VL** RunPod template:

- **Docker image**: `huchukato/comfyui-qwenvl-runpod:cu13-minimax`
- **Base**: `runpod/comfyui:cuda13.0`
- All custom nodes pre-installed
- All 4 workflows auto-downloaded at boot
- Models auto-downloaded at first boot (~50 GB, persistent)
- ComfyUI v0.30.0+ forced at boot
- Sage Attention, FP16 accumulation, async offload
- TensorRT upscaling + RIFE interpolation

[📖 README & instructions](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/runpod/README_MiniMaxH3.md)

### ComfyUI Args (pre-configured)

```
--disable-auto-launch
--fast fp16_accumulation
--use-sage-attention
--reserve-vram 2
--cuda-malloc
--async-offload
```

---

## 🚀 Why Choose ComfyUI-QwenVL-Mod + MiniMax H3?

### 🎬 For Content Creators
- **Native audio**: Video and audio in one pass — no separate MMAudio needed
- **Multilingual**: Write in any language, Qwen3-VL handles translation
- **Professional**: Official MiniMax H3 prompt format with camera vocabulary and speaker tags
- **Quality**: 768p native, TensorRT upscale to higher resolution

### 🔥 For NSFW Content
- **Explicit**: Uncensored generation with dedicated NSFW presets
- **Detailed**: Rich scene descriptions with explicit diegetic soundscape
- **Natural**: Realistic progression, consistent characters
- **Audio**: Native moans, breaths, skin contact, ambient sounds

### ⚡ For Power Users
- **Customizable**: Easy to modify presets and system prompts
- **Extendable**: Add your own Qwen3-VL models (GGUF or HF)
- **Integrable**: Works with existing ComfyUI setups
- **Optimized**: Sage Attention, FP16, async offload, smart caching

---

## 🌟 What Makes This Special?

- **First**: Complete MiniMax H3 workflow pack with Qwen3-VL auto-prompting
- **Native audio**: No separate audio node — MiniMax H3 does it all
- **4 workflows**: T2VA, I2VA, FLF, R2VA — covers all MiniMax H3 modes
- **TensorRT**: Built-in upscaling and frame interpolation
- **NSFW**: Dedicated presets with explicit audio directives
- **Multilingual**: Any input language, auto-translated and formatted
- **Ready**: Works out-of-the-box with included workflows

---

## 🎯 What's New in v2.4

### 🚀 MiniMax H3 Support
- ✅ **4 workflows**: T2VA, I2VA (FL2VA/L2VA), FLF, R2VA
- ✅ **Native audio**: Video + stereo audio in one pass
- ✅ **NSFW presets**: 5s / 10s / 15s with explicit diegetic soundscape
- ✅ **Official format**: Three-field prompt (integrated_multimodal_description + overall_soundscape + non_diegetic_music)
- ✅ **Camera vocabulary**: Official MiniMax H3 camera motion terms
- ✅ **Speaker tags**: Dialogue with speaker IDs

### 🔍 Local Model Discovery
- ✅ Drop `.gguf` files in `models/LLM/GGUF` — auto-detected with `[local]` prefix
- ✅ Drop HF model dirs in `models/LLM/Qwen-VL` — auto-detected
- ✅ GGUF mmproj auto-pairing
- ✅ Multi-path support (respects `extra_model_paths.yaml`)

### 🧠 Qwen3.5 Support
- ✅ Architecture detection (reads GGUF header or HF config.json)
- ✅ Disable thinking mode for Qwen3.5
- ✅ Top K tuning (top_k=20 for Qwen3.5)
- ✅ New models: Qwen3.5-4B/9B/27B (uncensored, heretic, unsloth)

### ⚡ Performance
- ✅ SageAttention support
- ✅ FP16 accumulation
- ✅ Async offload
- ✅ Smart prompt caching with Fixed Seed Mode

---

## 📋 Credits

- **MiniMax H3** — [MiniMax](https://www.minimax.io/blog/minimax-h3) · [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **ComfyUI** — [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [PR #15224](https://github.com/Comfy-Org/ComfyUI/pull/15224)
- **QwenVL-Mod** — [huchukato/ComfyUI-QwenVL-Mod](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
- **Qwen3-VL** — [Qwen Team / Alibaba](https://github.com/QwenLM/Qwen3-VL)
- **TensorRT RIFE / Upscaler** — [huchukato](https://github.com/huchukato)
- **VideoHelperSuite** — [Kosinkadink](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- **Easy-Use** — [yolain](https://github.com/yolain/ComfyUI-Easy-Use)
- **was-node-suite** — [ltdrdata](https://github.com/ltdrdata/was-node-suite-comfyui)
- **find-perfect-resolution** — [ashtar1984](https://github.com/ashtar1984/comfyui-find-perfect-resolution)

---

## 📄 License

Workflows are released under the same license as the underlying models and custom nodes. See each repository for details.

MiniMax H3 model weights: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) — MiniMax H3 Community License.

---

Built with ❤️ for the ComfyUI community
