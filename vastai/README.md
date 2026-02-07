# ComfyUI - Wan 2.2 I2V/T2V - 5090 Blackwell

> **[Create an Instance](https://cloud.vast.ai/?ref_id=188688&creator_id=188688&name=ComfyUI%20-%20Wan%202.2%20I2V%2FT2V%20-%205090%20Blackwell)**

## What is this template?

This template gives you a **complete AI image and video generation environment** with ComfyUI running in a Docker container. ComfyUI provides a powerful node-based interface for creating complex AI workflows, from simple image generation to advanced multi-model pipelines.

**Think:** *"A visual programming environment for AI image and video generation - connect nodes to build any workflow you can imagine!"*

---

## What can I do with this?

### **🎬 WAN 2.2 Video Generation Workflows**
- **Text-to-Video (T2V)**: Generate professional 5-second videos from text prompts with cinematic timeline structure
- **Image-to-Video (I2V)**: Animate static images with automatic prompt enhancement and style detection
- **Extended Storyboard**: Create seamless storyboard-to-storyboard sequences with continuity focus
- **Cinematic Video**: Generate videos with professional cinematography specifications

### **🌐 Multilingual & Style Features**
- **Multilingual Support**: Process prompts from any language with automatic translation to English
- **Visual Style Detection**: Automatic detection and enhancement of artistic styles:
  - Anime, 3D cartoon, pixel art, puppet animation
  - 3D game style, claymation, watercolor
  - Black and white animation, oil painting
  - Felt style, tilt-shift photography, time-lapse photography

### **🧠 Smart Prompt System**
- **Auto-Prompt Enhancement**: Automatically enhance user prompts for optimal video generation
- **Professional Cinematography**: Built-in specifications for lighting, camera angles, shot types
- **Timeline Structure**: Precise 5-second timeline with frame-by-frame descriptions
- **Fixed Seed Mode**: Maintain consistent prompts regardless of media variations

### **⚡ Performance Features**
- **Smart Caching**: Automatic prompt caching prevents regeneration of identical prompts
- **GGUF Backend**: Efficient local model inference with quantization support
- **Hardware Optimization**: Automatic GPU detection and VRAM management
- **Batch Processing**: Generate multiple variations efficiently

### **🎯 Advanced Capabilities**
- **NSFW Support**: Comprehensive content generation without restrictions
- **Custom Presets**: Pre-configured workflows for different video styles
- **Prompt Enhancers**: Dedicated text-only prompt optimization nodes
- **Professional Output**: Industry-standard video generation quality

---

## Who is this for?

This is **perfect** if you:
- Want to use **QwenVL-Mod** with enhanced vision-language capabilities and smart prompt caching
- Need **professional WAN 2.2 video generation** with T2V and I2V workflows
- Want **multilingual support** with automatic language detection and translation
- Need **visual style detection** for anime, 3D, pixel art, and other artistic styles
- Want **Fixed Seed Mode** for consistent prompt generation regardless of media variations
- Need **WAN 2.2 I2V/T2V** support with timeline-based cinematic descriptions
- Want **Extended Storyboard** capabilities for seamless storyboard-to-storyboard generation
- Need **NSFW support** for comprehensive content generation
- Want **GGUF backend support** for efficient local model inference
- Need **prompt caching** for improved performance on repeated workflows
- Are creating **professional video content** with cinematography specifications
- Want **automated prompt enhancement** for optimal video generation results
- Need **hardware-optimized performance** with automatic GPU detection
- Want **industry-standard video quality** with professional output

---

## Quick Start Guide

### **Step 1: Configure Your Setup**
Set your preferred configuration via environment variables:
- **`WORKSPACE`**: Custom workspace directory for your models and outputs
- **`PROVISIONING_SCRIPT`**: URL to auto-download models on first boot
- **`HF_TOKEN`**: Hugging Face token for downloading restricted models (optional but recommended)
- **`CIVITAI_TOKEN`**: CivitAI token for premium model downloads (optional)

> **🎬 Pre-installed QwenVL-Mod Workflows**: This template includes complete workflows for WAN 2.2 video generation with QwenVL-Mod auto-prompt enhancement:
> - **WAN 2.2 T2V**: Text-to-video with automatic prompt optimization (HF & GGUF versions)
> - **WAN 2.2 I2V**: Image-to-video with style detection and enhancement (multiple variants)
> - **WAN 2.2 I2V-SVI**: Advanced image-to-video with SVI processing
> - **WAN 2.2 I2V-Full**: Complete workflow with MMAudio integration
> - **Extended Storyboard**: Seamless storyboard sequences with continuity
> - **Cinematic Video**: Professional cinematography specifications
> - **Multilingual Support**: Process prompts from any language
> - **Visual Style Detection**: Automatic detection of anime, 3D, pixel art styles
> - **SDXL LoRA Stack**: Professional image generation with upscaling
>
> **🔧 Auto-installed Dependencies**: The provisioning script automatically installs:
> - **llama-cpp-python**: Required for Qwen3-VL GGUF backend support
> - **All QwenVL-Mod dependencies**: Complete environment for vision-language processing
> - **WAN 2.2 model libraries**: Ready for video generation workflows
>
> **📦 Pre-installed Models**: Essential models for immediate workflow execution:
> - **WAN 2.2 VAE Models**: High-quality VAE for video generation
> - **SDXL VAE**: Professional image generation VAE
> - **ESRGAN Upscalers**: 2x upscaling models (standard & sharp variants)
> - **NSFW Text Encoder**: Uncensored text processing for WAN 2.2
> - **Complete model suite**: Ready for professional video and image generation

> **Template Customization:** Want to modify this setup? Click **edit**, make your changes, and save as your own template. Find it later in **"My Templates"**. [Full guide here](https://docs.vast.ai/templates)

### **Step 2: Launch Instance**
Click **"[Rent](https://cloud.vast.ai/?ref_id=188688)"** when you've found a suitable GPU instance

### **Step 3: Wait for Setup**
ComfyUI will be ready automatically with ComfyUI-Manager pre-installed *(initial model downloads may take additional time)*

### **Step 4: Access Your Environment**
**Easy access:** Click the **"Open"** button for instant access to the ComfyUI interface!

**Direct access via mapped ports:**
- **ComfyUI:** `http://your-instance-ip:8188` (main interface)

> **HTTPS Option:** Want secure connections? Set `ENABLE_HTTPS=true` in your Vast.ai account settings. You'll need to [install the Vast.ai certificate](https://docs.vast.ai/instances/jupyter) to avoid browser warnings.

### **Step 5: Start Creating**
Load a workflow, download models via ComfyUI-Manager, and start generating!

---

## Key Features

### **ComfyUI-Manager & Pre-installed Nodes**
- **One-click model downloads** from CivitAI and Hugging Face
- **Custom node installation** from the community repository
- **Workflow management** and sharing tools
- **Automatic dependency resolution** for custom nodes

> **🚀 Pre-installed Custom Nodes**: This template includes 30+ essential custom nodes for professional video generation:
> - **ComfyUI-Manager**: Essential node management system
> - **ComfyUI-QwenVL-Mod**: Enhanced vision-language with WAN 2.2 workflows
> - **ComfyUI-GGUF**: GGUF model support for efficient inference
> - **ComfyUI-VideoHelperSuite**: Video processing utilities
> - **ComfyUI-Impact-Pack**: Advanced node functionality
> - **ComfyUI_essentials**: Essential utilities and helpers
> - **Civicomfy**: CivitAI integration for model downloads
> - **was-node-suite-comfyui**: Comprehensive node suite
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
| **SSH Terminal** | System administration | [SSH access](https://docs.vast.ai/instances/sshscp) |
| **SSH Tunnel** | Bypass authentication | Forward local port to internal port 18188 |

### **Finding Your Credentials**
Access your authentication details:
```bash
# SSH into your instance
echo $OPEN_BUTTON_TOKEN          # For web access
```

*Use `https://` instead of `http://` if you've enabled HTTPS in your account settings.*

### **Port Reference**
| Service | External Port | Internal Port |
|---------|---------------|---------------|
| Instance Portal | 1111 | 11111 |
| ComfyUI | 8188 | 18188 |
| Jupyter | 8080 | 8080 |

### **Instance Portal (Application Manager)**
- **Service monitoring** with real-time status updates
- **Resource usage tracking** for GPU and memory
- **Log aggregation** for debugging issues
- **One-click service restarts** when needed

### **Advanced Features**

#### **Dynamic Provisioning**
Set `PROVISIONING_SCRIPT` environment variable to auto-download models and custom nodes from any public URL (GitHub, Gist, etc.)

#### **Multiple Access Methods**
| Method | Best For | Access Point |
|--------|----------|--------------|
| **Web Interface** | Workflow creation and generation | Port 8188 |
| **Jupyter Notebook** | Custom scripts and model management | `/jupyter` endpoint |
| **SSH Terminal** | File management and debugging | SSH connection |
| **SSH Tunnel** | Auth-free local access | Forward to port 18188 |

#### **Service Management**
```bash
# Check service status
supervisorctl status

# Restart ComfyUI
supervisorctl restart comfyui

# View service logs
supervisorctl tail -f comfyui
```

---

### **Model Organization**
ComfyUI organizes models in the following directories:
```
/workspace/ComfyUI/models/
├── checkpoints/     # Main model files (SD, FLUX, etc.)
├── loras/           # LoRA files
├── vae/             # VAE models
├── controlnet/      # ControlNet models
├── clip/            # CLIP models
├── embeddings/      # Textual embeddings
└── upscale_models/  # Upscaling models
```

**Note:** The Jupyter file browser cannot open the `checkpoints` folder directly due to a display limitation. Use the `ckpt` symlink instead, which points to the same directory. 

### **Environment Variables Reference**
| Variable | Default | Description |
|----------|---------|-------------|
| `WORKSPACE` | `/workspace` | ComfyUI workspace directory |
| `COMFYUI_ARGS` | `--disable-auto-launch --port 18188 --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload --supports-fp8-compute --fast cublas_ops autotune` | ComfyUI startup arguments optimized for RTX 5090 |
| `PROVISIONING_SCRIPT` | (none) | Auto-setup script URL |
| `HF_TOKEN` | (none) | Hugging Face token for restricted model downloads |
| `CIVITAI_TOKEN` | (none) | CivitAI token for premium model downloads |
| `ENABLE_HTTPS` | `false` | Enable HTTPS connections (set in Vast.ai account settings) |

### **Recommended GPU Memory**
| Use Case | Minimum VRAM | Recommended VRAM |
|----------|--------------|------------------|
| SD 1.5 / SDXL | 8 GB | 12 GB |
| FLUX.1 Dev | 16 GB | 24 GB |
| Video Generation | 24 GB | 48 GB+ |

### **CUDA Forward Compatibility**
Images tagged `cu130` or above automatically enable CUDA forward compatibility. This allows them to run on datacenter GPUs (e.g., H100, A100, L40S, RTX Pro series) with older driver versions. Consumer GPUs (e.g., RTX 4090, RTX 5090) do not support forward compatibility and require a driver version that natively supports CUDA 13.0 or above.

---

## Need More Help?

### **Documentation & Resources**
- **QwenVL-Mod Documentation:** [GitHub Repository](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
- **ComfyUI Documentation:** [Official Repository](https://github.com/Comfy-Org/ComfyUI)
- **ComfyUI-Manager:** [Node and Model Manager](https://github.com/Comfy-Org/ComfyUI-Manager)
- **Base Image Features:** [GitHub Repository](https://github.com/vast-ai/base-image/)
- **Instance Portal Guide:** [Vast.ai Instance Portal Documentation](https://docs.vast.ai/instance-portal)

### **Community & Support**
- **ComfyUI GitHub:** [Issues & Discussions](https://github.com/Comfy-Org/ComfyUI)
- **Vast.ai Support:** Use the messaging icon in the console

### **Getting Started Resources**
- **Example Workflows:** Pre-built workflows included in the image
- **ComfyUI Examples:** [Community workflow gallery](https://comfyworkflows.com/)
- **Custom Nodes:** Thousands available via ComfyUI-Manager

---

