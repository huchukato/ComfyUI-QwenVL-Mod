# ComfyUI-QwenVL Update Log

## Version 2.1.0 (2026/02/14)

🏷️ **User-Friendly Keep Last Prompt Feature**

This update refines the prompt persistence feature with a more intuitive parameter name and simplified implementation, making it easier for users to understand and use.

### 🏷️ Improved Parameter Naming
- **Renamed Parameter**: `bypass_mode` → `keep_last_prompt` for better user understanding
- **Clear Tooltip**: "Keep the last generated prompt instead of creating a new one"
- **Consistent Implementation**: Same behavior across all 6 nodes

### 🎯 Simplified Logic
- **Direct Memory Storage**: Uses simple `LAST_SAVED_PROMPT` variable instead of complex cache searching
- **Instant Response**: No file I/O or cache lookups - immediate prompt retrieval
- **Reliable Behavior**: Works regardless of seed changes or model variations

### ✅ Universal Implementation
- **All Nodes Updated**: QwenVL, QwenVL Advanced, QwenVL GGUF, QwenVL Advanced GGUF, QwenVL Prompt Enhancer, QwenVL Prompt Enhancer GGUF
- **Consistent UI**: Same parameter name and tooltip across all nodes
- **Bug Fixes**: Resolved TypeError issues in Advanced nodes

### 🔄 How Keep Last Prompt Works
1. **Generate Phase** (`keep_last_prompt=False`): Generate new prompt and save to memory
2. **Keep Phase** (`keep_last_prompt=True`): Return the last saved prompt from memory
3. **First Run**: If no previous prompt exists, returns empty string
4. **Memory Update**: Each generation overwrites the previous saved prompt

### 🐛 Bug Fixes
- **Fixed TypeError**: Resolved "unexpected keyword argument" errors in Advanced nodes
- **Parameter Consistency**: All process() functions now use `keep_last_prompt`
- **Syntax Errors**: Fixed duplicate global declarations

---

## Version 2.0.9 (2026/02/12)

🎛️ **Bypass Mode Parameter for Prompt Persistence**

This major feature introduces a new `bypass_mode` parameter that allows users to maintain previously generated prompts without regeneration, providing perfect workflow control.

### 🎛️ New Bypass Mode Feature
- **Smart Cache Retrieval**: When `bypass_mode=True`, nodes automatically retrieve the most recent cached prompt for the current model
- **Zero Resource Usage**: Bypass mode consumes no computational resources - instant response
- **Perfect Workflow Control**: Generate prompts once, then enable bypass mode to preserve them while changing inputs
- **Universal Implementation**: Available across all nodes (HF, GGUF, PromptEnhancer, Advanced variants)

### 🔄 How Bypass Mode Works
1. **Generate Phase** (`bypass_mode=False`): Generate new prompts and save them to cache
2. **Bypass Phase** (`bypass_mode=True`): Retrieve and return the most recent cached prompt
3. **Input Changes**: When bypass mode is enabled, changing images/videos or other inputs has no effect on the output
4. **Seamless Switching**: Toggle bypass mode on/off to switch between generation and persistence

### 📋 Updated Nodes
- ✅ **AILab_QwenVL** (HF): Added bypass_mode parameter
- ✅ **AILab_QwenVL_GGUF**: Added bypass_mode parameter  
- ✅ **AILab_QwenVL_GGUF_Advanced**: Added bypass_mode parameter
- ✅ **AILab_QwenVL_PromptEnhancer** (HF): Added bypass_mode parameter
- ✅ **AILab_QwenVL_GGUF_PromptEnhancer**: Added bypass_mode parameter

### 🎯 Use Cases
- **Prompt Locking**: Generate the perfect prompt once, then lock it while experimenting with other workflow parameters
- **Batch Processing**: Generate multiple prompts, then use bypass mode to maintain consistency across batches
- **Resource Optimization**: Enable bypass mode when you need to preserve prompts but don't want to consume GPU resources
- **Workflow Debugging**: Test other parts of your workflow without changing the generated prompts

### 🎮 Simple Control
- **Checkbox Interface**: Simple bypass_mode checkbox in all node interfaces
- **Default Behavior**: bypass_mode defaults to False (normal generation)
- **Instant Feedback**: Console logs show when bypass mode is active and which cached prompt is being used

### 🔧 Technical Implementation
- **Cache-Based**: Uses existing prompt caching system for reliable prompt retrieval
- **Model-Specific**: Retrieves prompts specific to the current model configuration
- **Fallback Handling**: If no cached prompt exists, returns empty string gracefully
- **Performance Optimized**: Minimal overhead when bypass mode is enabled

## Version 2.0.8 (2026/02/06)

🐛 **Bug Fixes and Stability Improvements**

This release addresses critical issues identified in v2.0.7 and enhances multilingual capabilities.

### 🔧 Bug Fixes
- **JSON Syntax Error**: Fixed trailing comma in `AILab_System_Prompts.json` at line 26
- **Undefined Variable Error**: Resolved `name 'image_hash' is not defined` in GGUF Advanced node
- **Fixed Seed Stability**: Reverted problematic fixed seed improvements that caused undefined variable errors
- **Node Loading**: Ensured all nodes load correctly without missing variable errors

### 🌐 Multilingual Support Enhancement
- **Complete Multilingual Support**: All WAN 2.2 presets now support "user prompts from any language"
- **Updated Presets**: 
  - 🍿 Wan 2.2 I2V: Multilingual + style detection
  - 🍿 Wan Extended Storyboard: Multilingual + style detection  
  - 🎥 Wan Cinematic Video: Multilingual + style detection
  - 🍿 Wan 2.2 T2V: Multilingual + style detection

### 🎨 Visual Style Detection
- **Enhanced Style Detection**: Comprehensive support for artistic styles:
  - Anime style
  - 3D cartoon
  - Pixel art
  - Puppet animation
  - 3D game style
  - Claymation
  - Watercolor
  - Black and white animation
  - Oil painting style
  - Felt style
  - Tilt-shift photography
  - Time-lapse photography

### 🔧 Technical Improvements
- **JSON Validation**: All system prompts now validate correctly
- **Error Handling**: Improved error messages for debugging
- **Code Stability**: Removed complex seed tracking that caused undefined variables
- **Documentation**: Updated README and changelog with comprehensive changes

### 🔄 Repository Management
- **Clean Revert**: Properly reverted problematic commits
- **Version Control**: Maintained stable commit history
- **Documentation**: Comprehensive update logs for transparency

---

## Version 2.0.7 (2026/02/04)

🧠 **Smart Prompt Caching System**

This release introduces intelligent prompt caching with Fixed Seed Mode for dramatic performance improvements and workflow stability.

### 🎯 Key Features
- **Smart Caching**: Automatic prompt caching prevents regeneration of identical prompts
- **Fixed Seed Mode**: Set any fixed seed value to maintain consistent prompts regardless of media variations
- **Universal Coverage**: Works across all node types (HF, GGUF, PromptEnhancer)
- **Performance Boost**: Instant response for cached prompts with zero model loading time
- **Simplified Logic**: Streamlined caching system that always includes seed for predictable behavior across all seed values

### 🔧 Technical Implementation
- **Hash-based Cache Keys**: MD5 hashes generated from model, preset, custom prompt, media, and seed
- **Media Hash Sampling**: Efficient image/video tensor hashing using pixel sampling
- **Cache File Storage**: JSON-based cache with automatic save/load functionality
- **Seed-based Logic**: Always includes seed in cache key for consistent behavior
- **Console Logging**: Clear feedback on cache hits/misses with seed information

### 📋 Cache Behavior
- **Fixed Seed Mode**: Any fixed seed value maintains consistent prompts regardless of media variations
- **Random Mode**: Different seeds generate different prompts even with identical inputs
- **Cache Hit**: Same seed + same inputs = instant response from cache
- **Cache Miss**: Different seed or different inputs = new generation and cache storage
- **Cache Persistence**: Automatic saving to `prompt_cache.json` after each generation
- **Cache Loading**: Automatic cache restoration on ComfyUI startup
- **Memory Efficient**: Cache only stores generated text and metadata

### 🎨 User Experience
- **Updated Tooltips**: Enhanced seed parameter tooltips explaining cache behavior
- **Instant Response**: Zero wait time for cached prompts
- **Workflow Stability**: Same prompt guaranteed with fixed seed regardless of media
- **Performance Gains**: Dramatic speedup for repeated prompt generation
- **Transparent Operation**: Clear console messages indicating cache usage

### 🛠️ Compatibility
- **Backward Compatible**: Existing workflows continue to work unchanged
- **Optional Feature**: Fixed Seed Mode works with any seed value
- **Cache Management**: Automatic cache cleanup and management
- **Cross-Session**: Cache persists between ComfyUI restarts
- **Code Maintenance**: Removed deprecated parameters across all download functions (HF, GGUF, PromptEnhancer) for future compatibility
- **GGUF Performance**: Increased default context size from 8192 to 32768 across all GGUF models for better utilization
- **Universal Caching**: Fixed Seed Mode and smart caching now available across all node types (HF, GGUF, PromptEnhancer) with shared cache file
- **Simplified Logic**: Streamlined caching system that always includes seed for predictable behavior across all seed values

---

## Version 2.0.6 (2026/02/03)

🎬 **Professional Cinematography Enhancement**

This release adds comprehensive professional cinematography specifications to all WAN 2.2 presets and updates naming for consistency.

### 🎯 Major Enhancements
- **Professional Specifications**: All WAN 2.2 presets now include comprehensive cinematography technical details
- **Light Sources**: 8 types (sunlight, artificial, moonlight, practical, firelight, fluorescent, cloudy, mixed)
- **Light Quality**: 9 qualities (soft, hard, top, side, back, bottom, edge, silhouette, low/high contrast)
- **Time Periods**: 6 periods (daytime, nighttime, dusk, sunset, dawn, sunrise)
- **Shot Types**: 6 types (close-up, medium shot, medium close-up, medium full shot, full shot, wide angle)
- **Composition Types**: 5 types (central, balanced, side, symmetrical, short side)
- **Lens Specifications**: 5 focal lengths + 6 angles for complete camera control
- **Camera Movements**: 9 movement types for dynamic cinematography
- **Color Tone**: 4 tone options (warm, cool, high saturation, low saturation)

### 🏷️ Branding Updates
- **Wan 2.2 T2V**: Updated to "🍿 Wan 2.2 T2V" with proper emoji
- **Extended Storyboard**: Updated to "🍿 Wan Extended Storyboard" for WAN family consistency
- **Cinematic Video**: Updated to "🎥 Wan Cinematic Video" to align with WAN branding

### 📋 Enhanced Presets
- **🍿 Wan 2.2 I2V**: Timeline structure + professional specs
- **🍿 Wan 2.2 T2V**: Timeline structure + professional specs  
- **🍿 Wan Extended Storyboard**: Timeline + continuity + professional specs
- **🎥 Wan Cinematic Video**: Single scene + professional specs

### 🛠️ Technical Improvements
- **Consistent Organization**: All WAN presets now follow family branding
- **Professional Quality**: Industry-standard cinematography specifications
- **Comprehensive Coverage**: Complete technical details for professional video generation

---

## Version 2.0.5 (2026/02/01)

🎬 **Extended Storyboard Preset Added**

This release adds a new specialized preset for extended storyboard generation with WAN 2.2 format compatibility.

### 🎯 New Features
- **Extended Storyboard Preset**: New preset positioned as 2nd in the list after Wan 2.2 I2V
- **WAN 2.2 Format Compliance**: Follows the same timeline structure "(At 0 seconds: ...)" format
- **Storyboard Continuity**: Each paragraph repeats content from previous paragraph for smooth transitions
- **NSFW Support**: Complete NSFW description support like Wan 2.2 I2V preset
- **Cinematic Specifications**: Comprehensive camera, lighting, and composition guidelines

### 📋 Preset Details
- **Position**: 2nd preset (after Wan 2.2 I2V) for easy access
- **Format**: 5-second timeline with detailed second-by-second descriptions
- **Focus**: Extended video generation with consistent character appearance and scene continuity
- **Technical Specs**: Includes camera movements, lighting types, composition guidelines
- **Output**: Optimized for storyboard-to-storyboard workflow generation

### 🛠️ Technical Improvements
- **Stability Fixes**: Identified and resolved "keep model loaded" memory issues
- **Performance**: Maintained stable operation without OOM errors
- **Compatibility**: Preset works with both VL and text-only models

---

## Version 2.0.4 (2026/02/01)

🔧 **Stability Update - SageAttention Removal**

This release focuses on stability by removing SageAttention integration that was causing compatibility issues and model interference.

### 🛠️ Major Changes
- **SageAttention Removed**: Completely removed SageAttention integration due to compatibility and stability issues
- **Simplified Attention Modes**: Now supports auto, flash_attention_2, and sdpa only
- **Clean Codebase**: Removed 100+ lines of complex patching code
- **Stable Performance**: SDPA provides excellent performance with zero interference

### 🎯 Attention Modes Available
- **Auto**: Automatically chooses best available attention implementation
- **Flash Attention 2**: High-performance attention when available (RTX 20+)
- **SDPA**: Scaled Dot Product Attention - stable and well-tested fallback

### 🐛 Issues Resolved
- **Model Output Problems**: Fixed prompt generation issues caused by SageAttention interference
- **Recursion Errors**: Eliminated infinite recursion in attention patching
- **Head Dimension Compatibility**: Removed headdim restrictions that were causing crashes
- **VAE Compatibility**: Fixed VAE encoding/decoding interference

### ⚡ Performance Impact
- **Flash Attention 2**: Still available for 2-3x speedup on compatible hardware
- **SDPA**: Highly optimized baseline performance
- **Zero Interference**: Clean attention pipeline without patching complications
- **Better Stability**: More reliable across different model architectures

---

## Version 2.0.3 (2026/02/01)

🔧 **SageAttention Compatibility Fix**

This hotfix release addresses a critical compatibility issue with SageAttention integration that prevented proper patching in certain transformer versions.

### 🐛 Bug Fixes
- **SageAttention Patch**: Fixed AttributeError when accessing `transformers.models.qwen2.modeling_qwen2.F`
- **Proper Module Access**: Changed patch target to `torch.nn.functional.F.scaled_dot_product_attention`
- **Compatibility**: Ensured SageAttention works across different transformer versions
- **Safer Imports**: Added better error handling for module imports

### ⚡ Performance Impact
- SageAttention now works correctly with 8-bit quantization
- Maintains 2-5x performance boost on compatible hardware
- Graceful fallback to SDPA when patching fails

---

## Version 2.0.2 (2026/02/01)

🎯 **User Experience & Model Management Update**

This release focuses on improving user accessibility, model management, and content generation capabilities based on community feedback.

### 🚀 Enhanced Model Accessibility
- **Free Abliterated Models**: Added `Qwen3-4B-abliterated-TIES` and `Qwen3-8B-abliterated-TIES` (nbeerbower) as default options
- **No Token Required**: These models provide uncensored performance without Hugging Face authentication
- **Smart Model Ordering**: 4B models prioritized before 8B for better VRAM accessibility
- **8-Bit Quantization Default**: Optimized VRAM usage for both VL and text models

**New Default Models:**
- **VL**: `Qwen3-VL-4B-Instruct-Abliterated` (8-bit quantized)
- **Text**: `Qwen3-4B-abliterated-TIES` (8-bit quantized, no token)

### 🔧 Improved Custom Prompt Logic
- **Template Combination**: Fixed custom prompts to combine with preset templates instead of replacing them
- **Consistent Behavior**: All nodes (HF, GGUF, PromptEnhancer) now handle custom prompts identically
- **Better UX**: Users can now add their input while keeping preset instructions intact
- **WAN 2.2 Compatibility**: Fixed I2V preset to work properly with custom user input

### 📝 Enhanced NSFW Content Generation
- **Comprehensive Descriptions**: Expanded NSFW rules to include detailed sexual act descriptions
- **Clear Instructions**: Added "body parts and acts being performed" specification
- **Complete Coverage**: Includes blowjob, paizuri, deepthroat, handjob, cunnilingus, anal, missionary, doggystyle, cowgirl, reverse cowgirl, masturbation, fingering, squirting, cumshot, facial
- **Applied Universally**: Enhanced both WAN 2.2 I2V and T2V presets

### 🎬 Preset Priority Optimization
- **WAN 2.2 I2V First**: Moved most-used video generation preset to top position
- **Better Workflow**: Faster access to primary video generation functionality
- **Maintained Order**: Preserved logical organization of remaining presets

### 🐛 Bug Fixes
- **Custom Prompt Override**: Fixed issue where custom prompts completely replaced preset templates
- **GGUF Consistency**: Applied same prompt logic fixes to GGUF nodes
- **Tooltip Updates**: Updated all custom prompt tooltips to reflect new behavior
- **JSON Validation**: Removed duplicate model entries for cleaner configuration

---

## Version 2.0.1-enhanced (2026/01/30)

🚀 **Major Performance & Video Generation Update**

This enhanced version introduces cutting-edge performance optimizations and comprehensive video generation capabilities, making it the most advanced QwenVL integration available.

### ⚡ SageAttention Integration
- **2-5x Performance Boost**: Added SageAttention support for 8-bit quantized attention computation
- **Automatic Detection**: System automatically detects SageAttention availability and GPU compatibility
- **Seamless Fallback**: Gracefully falls back to SDPA when SageAttention is unavailable
- **Memory Efficiency**: Reduced VRAM usage with quantized attention kernels

**Requirements:**
- NVIDIA GPU with capability >= 8.0 (RTX 30/40/50 series)
- CUDA >= 12.0
- sageattention package (optional)

### 🎬 WAN 2.2 Video Generation Integration
- **I2V Support**: Image-to-Video generation with cinematic timeline structure
- **T2V Support**: Text-to-Video generation with professional scene descriptions
- **5-Second Timeline**: Precise second-by-second scene progression
- **Multilingual**: Italian/English input to optimized English output
- **Cinematic Quality**: Film-style direction including lighting, camera, composition

**Available Prompts:**
- `🍿 Wan 2.2 I2V` - For image-to-video workflows in QwenVL nodes
- `🍿 Wan 2.2 T2V` - For text-to-video workflows in Prompt Enhancer nodes

### 🔧 Enhanced Compatibility
- **Transformers Compatibility**: Support for both old (`AutoModelForImageTextToText`) and new (`AutoModelForVision2Seq`) transformer versions
- **Attention Mode Selection**: Four modes available - auto, flash_attention_2, sdpa, sageattention
- **Custom Model Support**: Extended HF and GGUF model configurations
- **Improved Error Handling**: Better diagnostics and fallback mechanisms

### 📚 Documentation Updates
- **SageAttention Section**: Comprehensive installation and troubleshooting guide
- **WAN 2.2 Documentation**: Complete usage examples and best practices
- **Performance Comparisons**: Detailed benchmarks and requirements tables
- **Updated README**: Professional documentation with all new features

### 🏷️ Manager Integration
- **Custom Nodes JSON**: Prepared for ComfyUI Manager submission
- **Enhanced Tags**: Added wan2.2, video, i2v, t2v tags for better discoverability
- **Updated Metadata**: Comprehensive description highlighting all enhancements

---

# Release Notes: v2.0.0 (2025-12-22)

### New GGUF Nodes
We've expanded our GGUF model support with three powerful new nodes:

- **QwenVL (GGUF)** — Lightweight GGUF-based vision node for image/video understanding and text generation. Offers significantly faster inference speed compared to Transformers models, making it ideal for real-time workflows and resource-constrained environments.
- **QwenVL (GGUF) Advanced** — Enhanced GGUF vision node with additional controls for advanced users. Maintains the ultra-fast inference speed of GGUF while providing fine-tuned control over generation parameters.
- **Qwen Prompt Enhancer (GGUF)** — GGUF text-only node for intelligent prompt rewriting and enhancement (not a vision model). Delivers rapid prompt enhancement with minimal resource usage, perfect for iterative prompt refinement workflows.
- **Qwen Prompt Enhancer (Transformers)** — Uses Qwen3 transformer model to enhance and rewrite prompts. Analyzes your input prompt and intelligently expands it with better detail, structure, and clarity for improved generation quality. Offers full model capabilities with precise control over the enhancement process.

![Qwen V2.0.0](https://github.com/user-attachments/assets/5187b98c-ccb5-4b57-858d-e43fbfb04a98)

### Enhanced GPU Device Selection (Advanced Node)
The **QwenVL (Advanced)** node now offers flexible GPU device management:

- **Manual GPU selection** — Choose specific CUDA devices (e.g., `cuda:1`, `cuda:2`) instead of defaulting to `cuda:0`
- **Automatic device detection** — Dynamically discovers all available CUDA devices on your system
- **Improved device mapping** — More consistent behavior and better resource allocation
- **OOM prevention** — Route models to underutilized GPUs when your primary GPU is handling diffusion workloads

*Note: The basic QwenVL node continues to use automatic device selection for simplicity*

---

## ✨ Improvements

### GGUF: Quality and Usability

**Cleaner Outputs by Default:**
- Automatically removes common "thinking/planning" content and leaked tokens (`<think>`, `<im_start>`, `<im_end>`)
- Users now receive clean, usable prompt-only or answer-only text without manual filtering

**QwenVL (GGUF) Vision Node:**
- Model dropdown now displays actual `.gguf` filenames with automatic deduplication for easier model identification
- Enhanced download progress logging with clear status messages during model download and cache reuse
- Token generation speed reporting (`tok/s`) when available — helps compare different models and quantization levels

![QwenVL (GGUF) Vision Node](https://github.com/user-attachments/assets/bc9450d9-1695-452d-9e46-f05a4bf315de)

**Qwen Prompt Enhancer (GGUF):**
- Updated built-in presets to reduce "junk talk" and return clean enhanced prompts more consistently
- Refined system prompts that minimize verbose output
- More reliable prompt-only text generation

![Qwen Prompt Enhancer (GGUF)](https://github.com/user-attachments/assets/d809f6fa-b43f-40c3-89e1-d03dc5fa7dee)

### Transformers Nodes: GPU and Attention Stability


**Advanced GPU Routing:**
- `QwenVL (Advanced)` supports selecting specific GPUs (e.g., `cuda:1`, `cuda:2`) to avoid OOM when GPU0 is busy
- Improved device-mapping logic for more consistent behavior across different hardware configurations

**Attention Backend Stability:**
- Flash-Attention auto mode now behaves safely across all platforms
- Gracefully falls back to SDPA when Flash-Attention dependencies are unavailable
- Prevents runtime errors from missing or incompatible Flash-Attention installations

---

## 🐛 Bug Fixes

### QwenVL (Transformers) Stability
- **Fixed:** Invalid CUDA device handling that caused crashes with incorrect device specifications (e.g., device `"0"` or malformed `device_map`)  
  Related issue: https://github.com/1038lab/ComfyUI-QwenVL/issues/21
- **Fixed:** Flash-Attention detection now restricted to Linux systems only, preventing Windows metadata errors
- **Fixed:** Flash-Attention auto mode fallback mechanism to eliminate runtime errors when dependencies are unavailable

---

## 📚 Documentation & Dependencies

### New Documentation
- **Added:** Comprehensive installation guide for vision-capable `llama-cpp-python`  
  See `docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md` for:
  - JamePeng fork wheel installation instructions (wheel source: https://github.com/JamePeng/llama-cpp-python/releases/)
  - Handler verification steps
  - Common numpy/OpenCV conflict resolution 

### Dependencies
- **Added:** `hf_xet` to `requirements.txt` for improved Hugging Face download performance and to eliminate Xet fallback warnings
## Version 1.1.0 (2025/11/11)

⚡ Major Performance Optimization Update

This release introduces a full rework of the QwenVL runtime to significantly improve speed, stability, and GPU utilization.

![QwenVL_V1.1.0](https://github.com/user-attachments/assets/13e89746-a04e-41a3-9026-7079b29e149c)

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
