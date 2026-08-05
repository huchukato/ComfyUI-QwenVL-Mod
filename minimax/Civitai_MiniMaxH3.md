# [MiniMax H3] NSFW T2VA/I2VA/FL2VA/R2VA Workflows 🎬 Auto Prompt | Native Audio | TensorRT Upscale | RIFE Interpolation

![MiniMax H3 Qwen3VL](https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/img/bannerminimax.png)

ComfyUI-QwenVL-Mod — Enhanced Vision-Language with MiniMax H3
Version 2.5 (2026/08/05) — 🎬 MiniMax H3 Native Video+Audio + Qwen3-VL Auto-Prompting

---

## ⚠️ Requirements — Read First!

### GPU & VRAM

- 🟢 **Recommended** — RTX 5090 / 4090 (24 GB+) → INT8 pruned → Fast, best quality
- 🟡 **Enthusiast** — RTX 3090 / 4080 (16-24 GB) → INT8 pruned + offload → Slower, good quality
- 🟠 **Budget** — RTX 3060 / 4060 Ti (12-16 GB) → INT4 pruned + offload → Slow, usable quality
- 🔴 **Experimental** — Blackwell GPUs (12 GB+) → NVFP4 → Requires Blackwell Tensor Cores

> **12 GB GPUs (e.g. RTX 3060 12GB)**: Technically possible with INT4 models + aggressive offloading, but **very slow**. You need 32 GB+ system RAM and a fast NVMe SSD. Not recommended for production use.

### Model Quantization Options

- **BF16 (full)** — Diffusion ~42 GB + Text encoder ~65 GB = ~110 GB total → [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **INT8 (pruned)** — Diffusion ~21 GB + Text encoder ~24.5 GB = ~50 GB total → [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **INT4 (pruned)** — Diffusion ~11 GB + Text encoder ~15 GB = ~24.5 GB total → [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)
- **NVFP4 (pruned)** — Diffusion ~12.5 GB + Text encoder ~15 GB = ~28 GB total → [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (Blackwell)

### Software

- **ComfyUI**: v0.30.0+ (required for MiniMax H3 native support)
- **Python**: 3.10+
- **CUDA**: 12.8+ (13.0 recommended)
- **Storage**: 30-110 GB SSD depending on quantization

### Qwen3-VL Prompt Enhancer

- **GGUF**: Q4_K_S (~4.8 GB) or Q5_K_S (~5.5 GB) for 8B model
- **HF**: Qwen3-VL-8B-Heretic-Stable (~16 GB) or Qwen3-VL-4B (~8 GB)

---

## 🌟 What is ComfyUI-QwenVL-Mod?

A powerful enhanced vision-language node for ComfyUI that combines **Qwen3-VL** models with **MiniMax H3** video generation workflows. Features multilingual support, visual style detection, native stereo audio, and NSFW capabilities for professional AI content creation.

Think: *"Your all-in-one solution for intelligent prompt enhancement and video+audio generation with MiniMax H3!"*

---

## 🎬 Key Features

### 🚀 MiniMax H3 Video+Audio Generation

- **T2VA** (Text-to-Video+Audio): Generate video with native stereo audio from text
- **I2VA** (Image-to-Video+Audio): Animate a first-frame image with audio
- **FL2VA** (First-Last-Frame): Generate the transition between two keyframes — Qwen3-VL sees both frames
- **R2VA** (Reference-to-Video): Lock character identity, style, motion, or voice using reference images

### 🧠 Qwen3-VL Auto-Prompting

- **Multilingual**: Write your prompt in **any language** — Qwen3-VL translates and converts it
- **Auto-format**: Generates the official MiniMax H3 prompt format (3-field for base, 6-field for R2VA)
- **Multi-reference**: Qwen3-VL sees all connected images via `image` + `image2` inputs
- **Visual style detection**: 12+ artistic styles (photorealistic, cinematic, anime, 3D CG, claymation, vintage film, watercolor, fantasy, etc.)
- **Smart caching**: Performance optimization with Fixed Seed Mode
- **GGUF backend**: Efficient local model inference with quantization support
- **Qwen3.5 support**: Thinking mode disabled via `/no_think` for fast prompt generation

### 🔊 Native Stereo Audio

- **No separate audio node needed** — MiniMax H3 generates video and audio jointly in a single forward pass
- Voice, sound effects, and music modeled together, not layered on afterward
- Describe sounds in your prompt and the model generates them natively

### 🎨 NSFW Support

- Comprehensive content generation without restrictions
- 9 dedicated NSFW presets (3 base 🎬 + 3 R2VA 🎞️ + 3 FL2VA 🔄) with explicit diegetic soundscape
- Natural progression, style adaptation, consistent characters

---

## 📦 What's Included — 4 Workflows

1. 📝 **T2VA** — `MiniMaxH3-T2VA-Qwen3VL.json` — text only — Text-to-video+audio. Simplest workflow. Uses PromptEnhancer (text-only).
2. 🖼️ **I2VA** — `MiniMaxH3-I2VA-Qwen3VL.json` — text + first-frame image (`image`) — Image-to-video. First-frame animation with audio.
3. 🔄 **FL2VA** — `MiniMaxH3-FL2VA-Qwen3VL.json` — text + first-frame (`image`) + last-frame (`image2`) — First-Last-Frame to video. Qwen3-VL sees both frames and describes the transition. Includes TensorRT upscale + RIFE frame interpolation for 48 fps output.
4. 🎞️ **R2VA** — `MiniMaxH3-R2VA-Qwen3VL.json` — text + reference images (`image` + `image2`) — Reference-to-video. Qwen3-VL sees all references. Lock identity, style, motion, camera, or voice using up to 9 ref images.

> Workflows 3 and 4 include **TensorRT upscaling** (RealESRGAN x4) and **RIFE frame interpolation** (rife49) for 48 fps high-resolution output.

---

## 🖼️ Multi-Reference Input (image2)

The QwenVL-Mod node has two image inputs:

- **T2VA**: no images needed
- **I2VA**: `image` = first frame
- **FL2VA**: `image` = first frame, `image2` = last frame, `frame_count` = 1
- **R2VA**: `image` = primary reference, `image2` = additional references (batch, up to 9), `frame_count` = 1–9

> Qwen3-VL sees **all** connected images as individual images (not as a video sequence), enabling proper multi-reference analysis for FL2VA and R2VA.

---

## 🎯 QwenVL-Mod NSFW Presets (9 total)

The workflows include built-in NSFW presets for the Qwen3-VL prompt enhancer:

### 🎬 Base Presets (T2VA / I2VA)

- `🎬 MiniMax H3 NSFW (5s)` — 5 seconds — 3 fields: `integrated_multimodal_description` + `overall_soundscape` + `non_diegetic_music`
- `🎬 MiniMax H3 NSFW (10s)` — 10 seconds — Same format
- `🎬 MiniMax H3 NSFW (15s)` — 15 seconds — Same format

### 🔄 FL2VA Presets (First-Last-Frame)

- `🔄 MiniMax H3 NSFW FL2VA (5s)` — 5 seconds — 3 fields, transition-focused (describes the path between frames)
- `🔄 MiniMax H3 NSFW FL2VA (10s)` — 10 seconds — Same format
- `🔄 MiniMax H3 NSFW FL2VA (15s)` — 15 seconds — Same format

### 🎞️ R2VA Presets (Reference)

- `🎞️ MiniMax H3 NSFW R2VA (5s)` — 5 seconds — 6 fields: `subject_definitions` + `summary` + `retention_analysis` + `detailed_description` + `overall_soundscape` + `non_diegetic_music`
- `🎞️ MiniMax H3 NSFW R2VA (10s)` — 10 seconds — Same format
- `🎞️ MiniMax H3 NSFW R2VA (15s)` — 15 seconds — Same format

### What the presets produce

- 🎬 **Base**: `[Shot 1]` with style + initial composition, camera vocabulary, speaker IDs, diegetic soundscape
- 🔄 **FL2VA**: Describes the **transition path** between first and last frames (not the scene — images fix the scene). Favors single continuous shot.
- 🎞️ **R2VA**: 6-section format with `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>` labels, retention markers (`fully_preserved`, `partially_preserved`, etc.), task-type summary
- All presets: **smooth, continuous camera motion** (no abrupt or stepped changes), explicit diegetic soundscape, optional non-diegetic music (defaults to N/A)

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
2. Upload your first-frame image to `image`
3. Select preset `🎬 MiniMax H3 NSFW (5s/10s/15s)`
4. Write what happens next (in any language)
5. Generate animated video with audio

### First-Last-Frame (FL2VA)
1. Load `MiniMaxH3-FL2VA-Qwen3VL.json`
2. Upload first-frame to `image`, last-frame to `image2`, set `frame_count=1`
3. Select preset `🔄 MiniMax H3 NSFW FL2VA (5s/10s/15s)`
4. Describe the transition between the two frames
5. Generate the interpolated video at 48 fps with TensorRT upscale + RIFE

### Reference-to-Video (R2VA)
1. Load `MiniMaxH3-R2VA-Qwen3VL.json`
2. Upload primary reference to `image`, additional references to `image2` (batch), set `frame_count` to match
3. Select preset `🎞️ MiniMax H3 NSFW R2VA (5s/10s/15s)`
4. Reference them by tag in your prompt: `<Picture 1>`, `<Picture 2>`, etc.
5. Generate video with locked identity/style

---

## 🔧 Technical Specifications

### ⚡ Performance

- **Output**: 768p, 24 fps (native), up to ~15 seconds
- **Audio**: Native stereo, generated jointly with video
- **Upscale**: TensorRT RealESRGAN x4 (FL2VA + R2VA workflows)
- **Frame interpolation**: RIFE rife49 → 48 fps (FL2VA + R2VA workflows)
- **Sage Attention**: FP16 accumulation, async offload
- **Smart caching**: Reuse prompts with same inputs, Fixed Seed Mode for text-only caching

### 🎨 Model Support

- **Qwen3-VL 4B**: 7 GGUF variants (2.38 GB – 4.28 GB)
- **Qwen3-VL 8B**: 7 GGUF variants (4.8 GB – 8.71 GB)
- **Qwen3.5**: 4B / 9B / 27B (uncensored, heretic, unsloth) — thinking mode disabled
- **HF Models**: Josiefed, official, Heretic-Stable variants
- **Quantization**: Q4_K_S, Q5_K_S, FP16, INT8

### 🌐 Multilingual Capabilities

- **Input languages**: Any language supported
- **Auto-translation**: Automatic translation to optimized English
- **Style detection**: Works with multilingual prompts
- **Cultural adaptation**: Context-aware prompt enhancement

---

## 📦 Installation

### Quick Install

1. Download: [ComfyUI-QwenVL-Mod](https://github.com/huchukato/ComfyUI-QwenVL-Mod) (latest version)
2. Extract to `ComfyUI/custom_nodes/ComfyUI-QwenVL-Mod`
3. Install requirements: `pip install -r requirements.txt`
4. Restart ComfyUI
5. Load included workflows from `minimax/` folder

### Custom Nodes Required

- **ComfyUI-QwenVL-Mod** — All workflows (Qwen3-VL prompt enhancer) — [huchukato/ComfyUI-QwenVL-Mod](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
- **ComfyUI-RIFE-TensorRT-Auto** — FL2VA, R2VA (frame interpolation) — [huchukato/ComfyUI-RIFE-TensorRT-Auto](https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto)
- **ComfyUI-Upscaler-TensorRT-Auto** — FL2VA, R2VA (upscaling) — [huchukato/ComfyUI-Upscaler-TensorRT-Auto](https://github.com/huchukato/ComfyUI-Upscaler-TensorRT-Auto)
- **ComfyUI-VideoHelperSuite** — FL2VA, R2VA (VHS_VideoCombine) — [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- **ComfyUI-Easy-Use** — FL2VA, R2VA (easy showAnything) — [yolain/ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
- **comfyui-find-perfect-resolution** — All workflows (ResolutionSelector) — [ashtar1984/comfyui-find-perfect-resolution](https://github.com/ashtar1984/comfyui-find-perfect-resolution)
- **was-node-suite-comfyui** — R2VA (ComfyMathExpression) — [ltdrdata/was-node-suite-comfyui](https://github.com/ltdrdata/was-node-suite-comfyui)

### Models Required

All MiniMax H3 models from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) on Hugging Face.

**T2VA / I2VA / FL2VA (fl2va)**

- `models/vae/` → `minimax_h3_video_vae_fp16.safetensors` (~5 GB)
- `models/vae/` → `minimax_h3_audio_vae_fp32.safetensors` (~0.6 GB)
- `models/diffusion_models/` → `10Eros_Max_H3_FL2VA-INT8-ConvRot.safetensors` (~21 GB)
- `models/text_encoders/` → `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` (~24.5 GB)

**R2VA (ref2va)** — same as above, except:

- `models/diffusion_models/` → `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (~21 GB)

**INT4 alternative** (for 12-16 GB GPUs): [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)

**Qwen3-VL Prompt Enhancer**

- `models/LLM/` → `Qwen3-VL-8B-Heretic-Stable` (GGUF or HF)

**TensorRT Engines (FL2VA + R2VA only)**

- `models/upscale_models/` → `RealESRGAN_x4` (TensorRT engine)
- `models/rife/` → `rife49_ensemble_True_scale_1_sim` (TensorRT engine)

> TensorRT engines must be built for your specific GPU. See [ComfyUI-RIFE-TensorRT-Auto](https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto) and [ComfyUI-Upscaler-TensorRT-Auto](https://github.com/huchukato/ComfyUI-Upscaler-TensorRT-Auto) for build instructions.

### Download Links

- **VAE**: [video_vae_fp16](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) · [audio_vae_fp32](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors)
- **Diffusion (fl2va)**: [10Eros_Max_H3_FL2VA-INT8-ConvRot.safetensors](https://huggingface.co/DmitryDB/MiniMax-H3-10Eros-Max-Quants/resolve/main/FL2VA/10Eros_Max_H3_FL2VA-INT8-ConvRot.safetensors)
- **Diffusion (ref2va)**: [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
- **Text encoder**: [qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot/resolve/main/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors)
- **INT4 models**: [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)

---

## 🎬 MiniMax H3 Prompting Notes

### How to Write Your Prompt

Describe the scene naturally. Be clear about the concepts below — Qwen3-VL handles the rest:

- **🎨 Visual style** (put it first): `photorealistic`, `cinematic`, `anime`, `3D CG`, `claymation`, `vintage film`, `watercolor`, `fantasy`
- **👥 Subjects**: number, gender, appearance, clothing, position, expression
- **🏃 Action / motion**: what happens, speed, interaction
- **🎥 Camera**: dolly, pan, zoom, static, handheld, crane, orbit — **smooth and continuous** (no abrupt changes)
- **🌍 Environment**: setting, lighting, atmosphere, time of day
- **🔊 Audio** (important!): dialogue, breaths, moans, skin contact, ambient sounds, music

> 🔄 **FL2VA**: Describe the **transition** between frames, not the scene (images fix the scene)
> 🎞️ **R2VA**: Reference inputs by tag: `<Picture 1>`, `<Picture 2>`, `<Video 1>`, `<Audio 1>`

### Resolution Guidance

MiniMax H3 native canvas: **768 px short edge**, long edge capped at **1344 px**, multiples of **32**.

- 📱 **Portrait**: 768×1344 · 896×1152 · 960×1280
- ⬛ **Square**: 1024×1024
- 🖥️ **Landscape**: 1344×768 · 1152×896 · 1280×960

> ⚠️ **Match the aspect ratio to your input image!** Forcing 16:9 on a portrait image will squash it.
>
> ⚠️ **Avoid direct 1080p.** Generate at native resolution, then upscale with TensorRT nodes (FL2VA + R2VA workflows).

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
- **9 presets**: 3 base 🎬 + 3 FL2VA 🔄 + 3 R2VA 🎞️ — each tuned for its mode
- **Detailed**: Rich scene descriptions with explicit diegetic soundscape
- **Natural**: Realistic progression, consistent characters
- **Audio**: Native moans, breaths, skin contact, ambient sounds

### ⚡ For Power Users
- **Customizable**: Easy to modify presets and system prompts
- **Extendable**: Add your own Qwen3-VL models (GGUF or HF)
- **Integrable**: Works with existing ComfyUI setups
- **Optimized**: Sage Attention, FP16, async offload, smart caching
- **Multi-reference**: `image2` input for FL2VA and R2VA workflows

---

## 🌟 What Makes This Special?

- **First**: Complete MiniMax H3 workflow pack with Qwen3-VL auto-prompting
- **Native audio**: No separate audio node — MiniMax H3 does it all
- **4 workflows**: T2VA, I2VA, FL2VA, R2VA — covers all MiniMax H3 modes
- **Multi-reference**: Qwen3-VL sees all connected images (not just the first)
- **TensorRT**: Built-in upscaling and frame interpolation
- **9 NSFW presets**: Dedicated presets for each mode with correct prompt structure
- **Multilingual**: Any input language, auto-translated and formatted
- **Ready**: Works out-of-the-box with included workflows

---

## 🎯 What's New in v2.5

### 🚀 MiniMax H3 Full Support
- ✅ **4 workflows**: T2VA, I2VA, FL2VA, R2VA — all modes covered
- ✅ **Multi-reference input**: `image2` input — Qwen3-VL sees all images as individual images
- ✅ **9 NSFW presets**: 3 base 🎬 + 3 R2VA 🎞️ + 3 FL2VA 🔄 with correct prompt structure
- ✅ **Smooth camera**: All presets enforce smooth, continuous camera motion
- ✅ **Native audio**: Video + stereo audio in one pass
- ✅ **Official format**: 3-field (base) and 6-field (R2VA) prompt formats

### � Qwen3.5 Thinking Fix
- ✅ `/no_think` prefix for Qwen3.5 models (enable_thinking deprecated in recent llama.cpp)
- ✅ Broadened architecture detection (qwen35, qwen35moe, qwen35_vl)
- ✅ Works across both HF and GGUF nodes

### 📦 Workflow Organization
- ✅ Moved workflows to `minimax/` folder
- ✅ Renamed FLF to FL2VA (clearer naming)
- ✅ Added Civitai documentation

---

## 📋 Credits

- **MiniMax H3** — [MiniMax](https://www.minimax.io/blog/minimax-h3) · [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **ComfyUI** — [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **QwenVL-Mod** — [huchukato/ComfyUI-QwenVL-Mod](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
- **Qwen3-VL** — [Qwen Team / Alibaba](https://github.com/QwenLM/Qwen3-VL)
- **INT4 models** — [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)
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
