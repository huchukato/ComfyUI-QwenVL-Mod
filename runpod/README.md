# ComfyUI-QwenVL-Mod RunPod Worker

Professional ComfyUI environment optimized for RunPod with QwenVL-Mod enhanced vision-language capabilities and WAN 2.2 video generation workflows.

**Clean template** - Add your own HF_TOKEN and CIVITAI_TOKEN when creating endpoints for restricted model access.

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
- **Cinematic Video**: Professional cinematography specifications

### **🌐 Enhanced Capabilities**
- **Multilingual Support**: Process prompts from any language
- **Visual Style Detection**: 12+ artistic styles (anime, 3D, pixel art, etc.)
- **Smart Prompt Caching**: Performance optimization with Fixed Seed Mode
- **GGUF Backend**: Efficient local model inference
- **NSFW Support**: Comprehensive content generation

### **⚡ Performance Optimizations**
- **RTX 5090 Optimized**: CUDA 13.0 with advanced optimizations
- **FP16 Accumulation**: Faster mixed precision computation
- **Sage Attention**: Optimized attention mechanisms
- **Async Offload**: Efficient memory management
- **FP8 Support**: Latest hardware acceleration

## 📦 Pre-installed Components

### **Custom Nodes (30+)**
- **ComfyUI-QwenVL-Mod**: Enhanced vision-language with WAN 2.2
- **ComfyUI-Manager**: Node and model management
- **ComfyUI-GGUF**: GGUF model support
- **ComfyUI-VideoHelperSuite**: Video processing utilities
- **ComfyUI-Impact-Pack**: Advanced functionality
- **PainterI2V Series**: Professional image-to-video workflows
- **ComfyUI-MMAudio**: Audio processing
- **ComfyUI-VFI**: Video frame interpolation
- And 20+ more essential nodes

### **Workflows Included**
> **🎬 Pre-installed QwenVL-Mod Workflows**: Complete WAN 2.2 video generation with auto-prompt enhancement:
- **WAN 2.2 T2V**: Text-to-video with automatic prompt optimization
- **WAN 2.2 I2V**: Image-to-video with style detection and enhancement
- **WAN 2.2 I2V-SVI**: Advanced image-to-video with SVI processing
- **SDXL LoRA Stack**: Professional image generation with upscaling
- **Extended Storyboard**: Seamless storyboard sequences with continuity
- **Multilingual Support**: Process prompts from any language
- **Visual Style Detection**: Automatic detection of anime, 3D, pixel art styles

### **Pre-loaded Models**
> **📦 Essential Models Included**: Ready for immediate workflow execution:
- **WAN 2.2 VAE**: High-quality video generation VAE
- **SDXL VAE**: Professional image generation VAE
- **ESRGAN Upscalers**: 2x upscaling models (standard & sharp variants)
- **NSFW Text Encoder**: Uncensored text processing for WAN 2.2
- **Complete model suite**: Ready for professional video and image generation

> **🔧 Auto-installed Dependencies**: The Docker image automatically includes:
> - **llama-cpp-python**: Required for Qwen3-VL GGUF backend support
> - **All QwenVL-Mod dependencies**: Complete environment for vision-language processing
> - **WAN 2.2 model libraries**: Ready for video generation workflows

## 🔧 Usage

### **Quick Start**
```bash
# Build the image
docker build -t huchukato/comfyui-qwenvl-runpod .

# Run locally (for testing)
docker run -p 8080:8080 --gpus all huchukato/comfyui-qwenvl-runpod

# Push to Docker Hub
docker push huchukato/comfyui-qwenvl-runpod
```

### **RunPod Deployment**
1. **[Create RunPod Account](https://runpod.io?ref_id=188688)** (Use referral ID: 188688)
2. Create a new RunPod endpoint using the template
3. Select GPU: RTX 5090 or equivalent  
4. **Optional**: Add `HF_TOKEN` and `CIVITAI_TOKEN` for restricted model access
5. Deploy and access via provided URL

> **🔑 Authentication**: Add your own tokens in environment variables when creating endpoints for access to premium/restricted models.

### **Environment Variables**
- `COMFYUI_ARGS`: ComfyUI startup arguments (pre-optimized)
- `PYTHONUNBUFFERED`: Python output buffering
- `HF_TOKEN`: Hugging Face token for restricted models (add when creating endpoint)
- `CIVITAI_TOKEN`: CivitAI token for premium models (add when creating endpoint)

> **💡 Note**: The template is provided without authentication tokens for security and flexibility. Add your own tokens when creating endpoints for access to restricted models.

## 🎯 Optimizations

### **GPU Optimizations**
```bash
--fast fp16_accumulation     # Faster FP16 operations
--use-sage-attention         # Optimized attention
--reserve-vram 2            # Smart VRAM management
--cuda-malloc               # CUDA memory optimization
--async-offload             # Async model offloading
--supports-fp8-compute      # FP8 acceleration
--fast cublas_ops autotune  # BLAS optimization
```

### **Memory Management**
- Automatic VRAM optimization for RTX 5090
- Efficient model loading and unloading
- Smart caching for repeated operations
- Async processing for better throughput

## 📋 System Requirements

### **Minimum**
- **GPU**: RTX 4090 (24GB VRAM)
- **CUDA**: 12.0+
- **Memory**: 32GB RAM
- **Storage**: 50GB available

### **Recommended**
- **GPU**: RTX 5090 (24GB VRAM)
- **CUDA**: 13.0+
- **Memory**: 64GB RAM
- **Storage**: 100GB available

## 🔗 Links

- **Repository**: https://github.com/huchukato/ComfyUI-QwenVL-Mod
- **Docker Hub**: https://hub.docker.com/r/huchukato/comfyui-qwenvl-runpod
- **Documentation**: https://github.com/huchukato/ComfyUI-QwenVL-Mod/tree/main/runpod

## 📝 Building from Source

### **Prerequisites**
- Docker with GPU support
- CUDA toolkit 13.0+
- Git

### **Build Commands**
```bash
# Clone repository
git clone https://github.com/huchukato/ComfyUI-QwenVL-Mod.git
cd ComfyUI-QwenVL-Mod/runpod

# Build image
docker build -t comfyui-qwenvl-runpod .

# Test locally
docker run -p 8080:8080 --gpus all comfyui-qwenvl-runpod
```

## 🐛 Troubleshooting

### **Common Issues**
1. **CUDA errors**: Ensure CUDA 13.0+ is installed
2. **Memory issues**: Check VRAM availability
3. **Model loading**: Verify internet connectivity for initial downloads

### **Health Check**
The container includes a health check that monitors ComfyUI availability:
```bash
curl -f http://localhost:8080/system_stats
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for the ComfyUI community**
