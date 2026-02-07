# ComfyUI-QwenVL-Mod RunPod Worker

Professional ComfyUI with QwenVL-Mod enhanced vision-language and WAN 2.2 video generation.

**Clean template** - Add HF_TOKEN and CIVITAI_TOKEN when creating endpoints.

---

## 🎯 Who is this for?

- AI Video Creators wanting WAN 2.2 video generation
- Content Creators needing multilingual prompt processing  
- Digital Artists requiring visual style detection
- ML Engineers deploying production workflows
- Researchers experimenting with vision-language models

---

## 🚀 Features

### **🎬 WAN 2.2 Video Generation**
- Text-to-Video (T2V): Professional 5-second video generation
- Image-to-Video (I2V): Advanced image animation with style detection
- Storyboard Workflows: Seamless storyboard-to-storyboard generation
- Cinematic Video: Professional cinematography specifications

### **🌐 Enhanced Capabilities**
- Multilingual Support: Process prompts from any language
- Visual Style Detection: 12+ artistic styles (anime, 3D, pixel art)
- Smart Prompt Caching: Performance optimization with Fixed Seed Mode
- GGUF Backend: Efficient local model inference
- NSFW Support: Comprehensive content generation

### **⚡ Performance Optimizations**
- RTX 5090/4090 optimized (CUDA 13.0, Python 3.12.12, PyTorch 2.10.0)
- FP16 accumulation, Sage Attention, Async Offload, FP8 support
- TensorRT support for upscaling and frame interpolation
- FileBrowser: Built-in file manager on port 8080
- JupyterLab: Terminal and development environment on port 8888

---

## 📦 Pre-installed Components

### **Custom Nodes (26)**
- ComfyUI-QwenVL-Mod: Enhanced vision-language with WAN 2.2
- ComfyUI-Manager: Node and model management
- ComfyUI-GGUF: GGUF model support
- ComfyUI-VideoHelperSuite: Video processing utilities
- ComfyUI-Impact-Pack: Advanced functionality
- ComfyUI-RunpodDirect: RunPod-specific optimizations
- ComfyUI-Upscaler-Tensorrt: TensorRT upscaling
- ComfyUI-RIFE-TensorRT-Auto: TensorRT frame interpolation
- And 18+ more essential nodes

### **🎬 Pre-installed Workflows (10)**
- WAN 2.2 T2V/I2V: Text/Image-to-video with auto prompt optimization
- WAN 2.2 I2V-SVI: Advanced image-to-video with SVI
- WAN 2.2 GGUF: Optimized versions for efficiency
- WAN 2.2 MMAudio: Audio-enhanced video generation
- SDXL LoRA Stack: Professional image generation with upscaling
- Extended Storyboard: Seamless storyboard sequences
- Multilingual Support: Process prompts from any language
- Visual Style Detection: Automatic detection of artistic styles

### **📦 Essential Models Included**
- WAN 2.2 VAE: High-quality video generation VAE
- SDXL VAE: Professional image generation VAE
- ESRGAN Upscalers: 2x upscaling models
- NSFW Text Encoder: Uncensored text processing for WAN 2.2

> **🔧 Dependencies**: llama-cpp-python for GGUF, QwenVL-Mod, WAN 2.2 libraries.

---

## 🔧 Usage

### **RunPod Deployment**
1. [Create RunPod Account](https://runpod.io?ref_id=188688) (Use referral ID: 188688)
2. Create new RunPod endpoint using the template
3. Select GPU: RTX 5090 or equivalent  
4. Optional: Add HF_TOKEN and CIVITAI_TOKEN for restricted model access
5. Deploy and access via provided URLs:
   - ComfyUI: main endpoint URL (port 8188)
   - FileBrowser: endpoint URL:8080 (no authentication required)
   - JupyterLab: endpoint URL:8888 (no token required)

> **🔑 Authentication**: Add your own tokens in environment variables when creating endpoints for access to premium/restricted models.

### **Environment Variables**
- COMFYUI_ARGS: ComfyUI startup arguments (pre-optimized)
- PYTHONUNBUFFERED: Python output buffering
- HF_TOKEN: Hugging Face token for restricted models (add when creating endpoint)
- CIVITAI_TOKEN: CivitAI token for premium models (add when creating endpoint)

> **💡 Note**: Template provided without authentication tokens for security.

---

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

---

## 📋 System Requirements

### **Minimum**
- GPU: RTX 4090 (24GB VRAM)
- CUDA: 12.0+
- Memory: 32GB RAM
- Storage: 50GB available

### **Recommended**
- GPU: RTX 5090 (24GB VRAM)
- CUDA: 13.0+
- Memory: 64GB RAM
- Storage: 100GB available

---

## 🔗 Links

- Repository: https://github.com/huchukato/ComfyUI-QwenVL-Mod
- Docker Hub: https://hub.docker.com/r/huchukato/comfyui-qwenvl-runpod

---

## 🐛 Troubleshooting

### **Common Issues**
1. CUDA errors: Ensure CUDA 13.0+ is installed
2. Memory issues: Check VRAM availability
3. Model loading: Verify internet connectivity for downloads

### **Health Check**
Container health check monitors ComfyUI:
```bash
curl -f http://0.0.0.0:8188/system_stats
```

---

**Built with ❤️ for the ComfyUI community**
