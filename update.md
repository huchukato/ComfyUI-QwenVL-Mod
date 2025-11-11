# ComfyUI-QwenVL Update Log
## Version 1.1.0 (2025/11/11)

⚡ Major Performance Optimization Update

This release introduces a full rework of the QwenVL runtime to significantly improve speed, stability, and GPU utilization.

### 🚀 Core Improvements
- **Flash Attention Integration (Auto Detection)**  
  Automatically leverages next-generation attention optimization for faster inference on supported GPUs, while falling back to SDPA when needed.
- **Attention Mode Selector**  
  Both QwenVL nodes expose the attention backend (auto / flash_attention_2 / sdpa) so users can quickly validate which mode performs best on their hardware without leaving the basic workflow view.
- **Precision Optimization**  
  Smarter internal precision handling improves throughput and keeps performance consistent across high-end and low-VRAM cards.
- **Runtime Acceleration**  
  The execution pipeline now keeps KV cache/device alignment always-on, cutting per-run overhead and reducing latency.
- **Caching System**  
  Models remain cached in memory between runs, drastically lowering reload times when prompts change.
- **Video Frame Optimization**  
  Streamlined frame sampling and preprocessing accelerate video-focused workflows.
- **Hardware Adaptation**  
  Smarter device detection ensures the best configuration across NVIDIA GPUs, Apple Silicon, and CPU fallback scenarios.

### 🧠 Developer Enhancements
- Unified model and processor loading with cleaner logging and fewer bottlenecks.  
- Refined quantization and memory handling for better stability across quant modes.  
- Improved fallback behavior when advanced GPU optimizations are unavailable.

### 💡 Compatibility
- Fully backward compatible with existing ComfyUI workflows.  
- Retains both **QwenVL** and **QwenVL (Advanced)** nodes: the basic node now bundles the most useful speed controls, while the advanced node exposes every knob (quantization, attention, device, torch.compile) for deep tuning.

### 🔧 Recommended
- PyTorch ≥ 2.8.0  
- CUDA 12.4 or later  
- Flash Attention 2.x (optional, for maximum performance)

> Switching quantization or attention modes forces a one-time model reload and is expected behavior when comparing runtime profiles.
### Version 1.0.4 (2025/10/31)

🆕 **Custom Model Support Added**
- Users can now add their own **custom Qwen-VL or Hugging Face models**  
  by creating a `custom_models.json` file in the plugin directory.  
  These models will automatically appear in the model selection list.

- Added automatic merging of user-defined models from `custom_models.json`,  
  following the same flexible mechanism as in *ComfyUI-JoyCaption*.

- Added detailed documentation  
  👉 [`docs/custom_models.md`](./docs/custom_models.md)  
  and an editable example file [`custom_models_example.json`](./custom_models_example.json).

⚙️ **Dependency Update**

- Updated **Transformers** version requirement:  
  `transformers>=4.57.0` (was `>=4.40.0`)  
  to ensure full compatibility with **Qwen3-VL** models.  
  [Reference: Qwen3-VL](https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#quickstart)

---
## Version 1.0.3 (2025/10/22)
- Added 8 more Qwen3-VL models 2B and 32B (FB16 and FP8 variants) have been integrated into our support list, catering to diverse requirements.

## Version 1.0.2 (2025/10/21)
- Integrated additional Qwen3-VL models
- Added Chinese language README (README_zh.md)
- Refined fine-tuning preset system prompt

## Version 1.0.1 (2025/10/17)
- Resolved various bugs
- Optimized video input logic

## v1.0.0 Initial Release (2025/10/17)
- Support for Qwen3-VL and Qwen2.5-VL series models.
- Automatic model downloading from Hugging Face.
- On-the-fly quantization (4-bit, 8-bit, FP16).
- Preset and Custom Prompt system for flexible and easy use.
- Includes both a standard and an advanced node for users of all levels.
- Hardware-aware safeguards for FP8 model compatibility.
- Image and Video (frame sequence) input support.
- "Keep Model Loaded" option for improved performance on sequential runs.
- Seed parameter for reproducible generation.
