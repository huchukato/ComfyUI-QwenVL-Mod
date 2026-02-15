# ComfyUI-QwenVL-Mod RunPod Worker - RTX 5090

Professional ComfyUI environment optimized for RunPod with QwenVL-Mod enhanced vision-language capabilities and WAN 2.2 video generation workflows.

**Template**: `OneClick-ComfyUI-WAN2.2-Qwen3VL-RTX5090` - **RTX 5090 Required**

---

## 🎯 Who is this for?

This template is perfect for:

- **AI Video Creators** wanting professional WAN 2.2 video generation
- **Content Creators** needing multilingual prompt processing
- **Digital Artists** requiring visual style detection and enhancement
- **ML Engineers** deploying production-ready video workflows
- **Researchers** experimenting with vision-language models
- **Professionals** needing NSFW-capable content generation

---

## 🚀 Features

### **🎬 WAN 2.2 Video Generation**
- **Text-to-Video (T2V)**: Professional 5-second video generation
- **Image-to-Video (I2V)**: Advanced image animation with style detection
- **Storyboard Workflows**: Seamless storyboard-to-storyboard generation
- **Story Generation**: 4-segment continuous video stories
- **Cinematic Video**: Professional cinematography specifications

### **🌐 Enhanced Capabilities**
- **Multilingual Support**: Process prompts from any language
- **Visual Style Detection**: 12+ artistic styles (anime, 3D, pixel art, etc.)
- **Smart Prompt Caching**: Performance optimization with Fixed Seed Mode
- **GGUF Backend**: Efficient local model inference
- **NSFW Support**: Comprehensive content generation
- **WebTerminal**: Direct bash access on port 8081

### **⚡ Performance Optimizations**
- **RTX 5090 Optimized**: CUDA 12.8 with advanced optimizations
- **FP16 Accumulation**: Faster mixed precision computation
- **Sage Attention**: Optimized attention mechanisms
- **Async Offload**: Efficient memory management
- **FP8 Support**: Latest hardware acceleration
- **TensorRT Support**: Advanced upscaling and interpolation

## 📦 Pre-installed Components

### **Custom Nodes (26)**
- **ComfyUI-QwenVL-Mod**: Enhanced vision-language with WAN 2.2
- **ComfyUI-Manager**: Node and model management
- **ComfyUI-GGUF**: GGUF model support
- **ComfyUI-VideoHelperSuite**: Video processing utilities
- **ComfyUI-Impact-Pack**: Advanced functionality
- **ComfyUI-RunpodDirect**: RunPod-specific optimizations
- **ComfyUI-Upscaler-Tensorrt**: TensorRT upscaling support
- **ComfyUI-RIFE-TensorRT-Auto**: TensorRT frame interpolation
- **PainterI2V Series**: Professional image-to-video workflows
- **ComfyUI-MMAudio**: Audio processing
- **ComfyUI-VFI**: Video frame interpolation
- And 15+ more essential nodes

### **Workflows Included (13)**
> **🎬 Pre-installed QwenVL-Mod Workflows**: Complete WAN 2.2 video generation with auto-prompt enhancement:
- **WAN 2.2 T2V**: Text-to-video with automatic prompt optimization
- **WAN 2.2 I2V**: Image-to-video with style detection and enhancement
- **WAN 2.2 T2V/I2V Story**: 4-segment continuous video stories
- **WAN 2.2 I2V-SVI**: Advanced image-to-video with SVI processing
- **WAN 2.2 T2V/I2V GGUF**: GGUF-optimized versions
- **WAN 2.2 MMAudio**: Audio-enhanced video generation
- **SDXL LoRA Stack**: Professional image generation with upscaling
- And 6+ more professional workflows

### **📦 Essential Models Included**
- **WAN 2.2 VAE**: High-quality video generation VAE
- **SDXL VAE**: Professional image generation VAE
- **ESRGAN Upscalers**: 2x upscaling models
- **NSFW Text Encoder**: Uncensored text processing for WAN 2.2

---

## 🔧 Usage

### **RunPod Deployment**
1. [Create RunPod Account](https://runpod.io?ref_id=188688) (Use referral ID: 188688)
2. Create new RunPod endpoint using template: `OneClick-ComfyUI-WAN2.2-Qwen3VL-RTX5090`
3. Select GPU: **RTX 5090** (required)
4. Optional: Add HF_TOKEN and CIVITAI_TOKEN for restricted model access
5. Deploy and access via provided URLs:
   - **ComfyUI**: main endpoint URL (port 8188)
   - **FileBrowser**: endpoint URL:8080 (no authentication required)
   - **JupyterLab**: endpoint URL:8888 (no token required)
   - **WebTerminal**: endpoint URL:8081 (direct bash access)

> **🔑 Authentication**: Add your own tokens in environment variables when creating endpoints for access to premium/restricted models.

---

## 🎯 Optimizations

### **RTX 5090 CUDA 12.8 Optimizations**
```bash
--fast fp16_accumulation     # Faster FP16 operations
--use-sage-attention         # Optimized attention
--reserve-vram 2            # Smart VRAM management
--cuda-malloc               # CUDA memory optimization
--async-offload             # Async model offloading
--supports-fp8-compute      # FP8 acceleration
--fast cublas_ops autotune  # BLAS optimization
```

---

## 📋 System Requirements

### **Minimum**
- **GPU**: RTX 5090 (24GB VRAM) - **REQUIRED**
- **CUDA**: 12.8 (included)
- **Memory**: 32GB RAM
- **Storage**: 50GB available

### **Recommended**
- **GPU**: RTX 5090 (24GB VRAM)
- **CUDA**: 12.8 (included)
- **Memory**: 64GB RAM
- **Storage**: 100GB available

---

## 🔗 Links

- **Repository**: https://github.com/huchukato/ComfyUI-QwenVL-Mod-5090
- **Docker Hub**: https://hub.docker.com/r/huchukato/comfyui-qwenvl-runpod-5090

---

## 🐛 Troubleshooting

### **Common Issues**
1. **CUDA errors**: Ensure RTX 5090 is selected (not RTX 4090)
2. **Memory issues**: Check VRAM availability
3. **Model loading**: Verify internet connectivity for downloads

### **Health Check**
Container health check monitors ComfyUI:
```bash
curl -f http://0.0.0.0:8188/system_stats
```

---

**Built with ❤️ for the ComfyUI community**
