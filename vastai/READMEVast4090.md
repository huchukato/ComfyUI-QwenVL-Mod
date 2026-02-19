# ComfyUI - WAN 2.2 I2V/T2V - RTX 4090

> **[Create an Instance](https://cloud.vast.ai/?ref_id=188688&creator_id=188688&name=ComfyUI%20-%20WAN%202.2%20I2V%2FT2V%20-%20RTX%204090)**

## What is this template?

This template gives you a **complete AI video generation environment** optimized for RTX 4090 with ComfyUI running in a Docker container. Features QwenVL-Mod enhanced vision-language capabilities and professional WAN 2.2 video generation workflows.

**Think:** *"A visual programming environment for AI video generation - connect nodes to build any workflow you can imagine!"*

---

## What can I do with this?

### **🎬 WAN 2.2 Video Generation Workflows**
- **Text-to-Video (T2V)**: Generate professional 5-second videos from text prompts
- **Image-to-Video (I2V)**: Animate static images with automatic prompt enhancement
- **Storyboard Workflows**: Seamless storyboard-to-storyboard generation
- **Story Generation**: 4-segment continuous video stories
- **Cinematic Video**: Professional cinematography specifications

### **🌐 QwenVL-Mod Enhanced Features**
- **Multilingual Support**: Process prompts from any language with automatic translation
- **Visual Style Detection**: 12+ artistic styles (anime, 3D, pixel art, etc.)
- **Smart Prompt Caching**: Performance optimization with Fixed Seed Mode
- **GGUF Backend**: Efficient local model inference with quantization support
- **NSFW Support**: Comprehensive content generation without restrictions

### **🧠 Smart Prompt System**
- **Auto-Prompt Enhancement**: Automatically enhance user prompts for optimal video generation
- **Professional Cinematography**: Built-in specifications for lighting, camera angles
- **Timeline Structure**: Precise 5-second timeline with frame-by-frame descriptions
- **Fixed Seed Mode**: Maintain consistent prompts regardless of media variations

### **⚡ RTX 4090 Optimized Performance**
- **SageAttention + FP16**: Proven optimal for WAN2.2 video generation
- **CUDA 12.4 Compatibility**: Maximum stability and performance
- **Miniconda Environment**: Python 3.12.12 isolated environment
- **Hardware Optimization**: Automatic GPU detection and VRAM management

---

## Who is this for?

This is **perfect** if you:
- Want to use **QwenVL-Mod** with enhanced vision-language capabilities
- Need **professional WAN 2.2 video generation** with T2V and I2V workflows
- Have an **RTX 4090** and want maximum performance optimization
- Want **multilingual support** with automatic language detection
- Need **visual style detection** for anime, 3D, pixel art styles
- Want **Fixed Seed Mode** for consistent prompt generation
- Need **NSFW support** for comprehensive content generation
- Want **GGUF backend support** for efficient local model inference
- Are creating **professional video content** with cinematography specs
- Want **hardware-optimized performance** for RTX 4090

---

## Quick Start Guide

### **Step 1: Configure Your Setup**
Set your preferred configuration via environment variables:
- **`WORKSPACE`**: Custom workspace directory for your models and outputs
- **`PROVISIONING_SCRIPT`**: URL to auto-download models on first boot
- **`HF_TOKEN`**: Hugging Face token for downloading restricted models
- **`CIVITAI_TOKEN`**: CivitAI token for premium model downloads

> **🎬 Pre-installed QwenVL-Mod Workflows**: Complete WAN 2.2 video generation with auto-prompt enhancement:
> - **WAN 2.2 T2V**: Text-to-video with automatic prompt optimization
> - **WAN 2.2 I2V**: Image-to-video with style detection and enhancement
> - **WAN 2.2 T2V/I2V Story**: 4-segment continuous video stories
> - **WAN 2.2 I2V-SVI**: Advanced image-to-video with SVI processing
> - **WAN 2.2 T2V/I2V GGUF**: GGUF-optimized versions
> - **WAN 2.2 MMAudio**: Audio-enhanced video generation
> - **Extended Storyboard**: Seamless storyboard sequences
> - **Multilingual Support**: Process prompts from any language
> - **Visual Style Detection**: Automatic detection of artistic styles

### **Step 2: Launch Instance**
Click **"[Rent](https://cloud.vast.ai/?ref_id=188688)"** when you've found a suitable RTX 4090 instance

### **Step 3: Wait for Setup**
ComfyUI will be ready automatically with all dependencies installed *(initial model downloads may take additional time)*

### **Step 4: Access Your Environment**
**Easy access:** Click the **"Open"** button for instant access to the ComfyUI interface!

**Direct access via mapped ports:**
- **ComfyUI:** `http://your-instance-ip:8188` (main interface)
- **JupyterLab:** `http://your-instance-ip:8888` (terminal access)

### **Step 5: Start Creating**
Load a workflow, download models via ComfyUI-Manager, and start generating!

---

## Key Features

### **ComfyUI-Manager & Pre-installed Nodes**
- **One-click model downloads** from CivitAI and Hugging Face
- **Custom node installation** from the community repository
- **Workflow management** and sharing tools
- **Automatic dependency resolution** for custom nodes

> **🚀 Pre-installed Custom Nodes**: 30+ essential custom nodes for professional video generation:
> - **ComfyUI-Manager**: Essential node management system
> - **ComfyUI-QwenVL-Mod**: Enhanced vision-language with WAN 2.2 workflows
> - **ComfyUI-GGUF**: GGUF model support for efficient inference
> - **ComfyUI-VideoHelperSuite**: Video processing utilities
> - **ComfyUI-Impact-Pack**: Advanced node functionality
> - **ComfyUI_essentials**: Essential utilities and helpers
> - **PainterI2V series**: Professional image-to-video workflows
> - **ComfyUI-WanMoeKSampler**: WAN model optimization
> - **ComfyUI-RIFE-TensorRT-Auto**: Advanced frame interpolation
> - **ComfyUI-MMAudio**: Audio processing capabilities
> - **ComfyUI-VFI**: Video frame interpolation
> - **Plus 20+ additional nodes** for professional workflows

### **Authentication & Access**
| Method | Use Case | Access Point |
|--------|----------|--------------|
| **Web Interface** | All ComfyUI operations | Click "Open" or port 8188 |
| **JupyterLab** | Terminal access & scripting | Port 8888 |
| **SSH Terminal** | System administration | [SSH access](https://docs.vast.ai/instances/sshscp) |

### **Port Reference**
| Service | External Port | Internal Port |
|---------|---------------|---------------|
| ComfyUI | 8188 | 18188 |
| JupyterLab | 8888 | 8080 |

---

## RTX 4090 Optimizations

### **Environment Variables Reference**
| Variable | Default | Description |
|----------|---------|-------------|
| `WORKSPACE` | `/workspace` | ComfyUI workspace directory |
| `COMFYUI_ARGS` | `--disable-auto-launch --port 18188 --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload` | ComfyUI startup arguments optimized for RTX 4090 |
| `PROVISIONING_SCRIPT` | (none) | Auto-setup script URL |
| `HF_TOKEN` | (none) | Hugging Face token for restricted model downloads |
| `CIVITAI_TOKEN` | (none) | CivitAI token for premium model downloads |

### **Why These Settings?**
- **SageAttention + FP16**: Proven optimal for WAN2.2 video generation
- **Reserve VRAM 2GB**: Perfect balance for RTX 4090 + WAN2.2
- **Conda Isolation**: Prevents dependency conflicts with Python 3.12.12
- **CUDA 12.4**: Maximum compatibility and stability for RTX 4090

### **Recommended GPU Memory**
| Use Case | Minimum VRAM | Recommended VRAM |
|----------|--------------|------------------|
| SD 1.5 / SDXL | 8 GB | 12 GB |
| WAN 2.2 Video | 16 GB | 24 GB |
| Professional Workflows | 24 GB | 24 GB+ |

---

## Need More Help?

### **Documentation & Resources**
- **QwenVL-Mod Documentation:** [GitHub Repository](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
- **ComfyUI Documentation:** [Official Repository](https://github.com/Comfy-Org/ComfyUI)
- **ComfyUI-Manager:** [Node and Model Manager](https://github.com/Comfy-Org/ComfyUI-Manager)
- **Vast.ai Documentation:** [Instance Portal Guide](https://docs.vast.ai/instance-portal)

### **Community & Support**
- **ComfyUI GitHub:** [Issues & Discussions](https://github.com/Comfy-Org/ComfyUI)
- **Vast.ai Support:** Use the messaging icon in the console

### **Getting Started Resources**
- **Example Workflows:** Pre-built WAN 2.2 workflows included
- **ComfyUI Examples:** [Community workflow gallery](https://comfyworkflows.com/)
- **Custom Nodes:** Thousands available via ComfyUI-Manager

---

**Built with ❤️ for the ComfyUI community - Optimized for RTX 4090**
