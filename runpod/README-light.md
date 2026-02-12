# ComfyUI-QwenVL-Mod (Light) - RunPod Deployment

## 🚀 Overview

**ComfyUI-QwenVL-Mod Light** - Optimized ComfyUI with Qwen3-VL support for RunPod.

### ✨ Features
- 🤖 Qwen3-VL vision-language model
- 🎬 WAN 2.2 Video Generation
- 🎨 SDXL Support
- 🌐 Multilingual (8+ languages)
- ⚡ GGUF Backend
- 🔧 RTX 5090 Optimized
- 📦 Fast startup

## 📋 Requirements

### Hardware
- **GPU**: RTX A4000+ (RTX 4090/5090 recommended)
- **VRAM**: 24GB+ minimum
- **Storage**: 50GB minimum
- **RAM**: 32GB+ recommended

## 🛠️ Quick Start

### 1. Deploy on RunPod

1. Go to [RunPod](https://runpod.io)
2. Deploy → Secure Cloud → GPU Instance
3. Configure:

```yaml
GPU: RTX A4000 (24GB) or higher
Storage: 50GB
Container Disk: 50GB
Image: huchukato/comfyui-qwenvl-cu13:light
Ports: 8188, 8080, 8888
```

### 2. Access Services

- **ComfyUI**: `https://<id>-8188.proxy.runpod.net`
- **FileBrowser**: `https://<id>-8080.proxy.runpod.net`
- **JupyterLab**: `https://<id>-8888.proxy.runpod.net`

## 🎯 Usage

### First Startup
Takes 5-10 minutes (downloads 25+ custom nodes).

### Workflows
- Text-to-Image: Prompt → Qwen2-VL → SDXL
- Image-to-Image: Image + Text → Qwen2-VL → SDXL
- Video Generation: Text/Image → Qwen2-VL → WAN 2.2

## 📁 File Structure

```
/ComfyUI/
├── custom_nodes/          # Auto-installed
├── models/
│   ├── vae/              # VAE models
│   ├── upscale_models/   # Upscale models
│   └── text_encoders/    # Text encoders
├── user/default/workflows/ # Pre-configured
├── input/                # Upload files
└── output/               # Generated content
```

## 🔧 Configuration

### Auto-installed Nodes
- ComfyUI-Manager, Essentials, GGUF
- Impact-Pack, RunpodDirect, Civicomfy
- Custom-Scripts, RIFE-TensorRT-Auto
- Translator, VideoHelperSuite
- Advanced-ControlNet, AnimateDiff-Evolved
- IP-Adapter, Easy-Use, KJNodes
- Frame-Interpolation, PainterI2V
- MMAudio, VFI, Upscaler-Tensorrt
- WanMoeKSampler

### Pre-installed Models
- WAN 2.1 VAE, SDXL VAE
- 2xLexicaRRDBNet (upscale)
- NSFW WAN UMT5-XXL (text encoder)

## 🔍 Troubleshooting

### Common Issues
1. **Slow First Startup**: Normal (5-10 min)
2. **Out of Memory**: Use smaller models
3. **Connection Issues**: Check RunPod status
4. **Model Downloads**: Restart container

### Performance Tips
- Use RTX 4090/5090
- Enable xformers
- Use GGUF models
- Clear cache regularly

##  Performance

| GPU | Text-to-Image | Video Generation |
|-----|---------------|-----------------|
| RTX A4000 | 2-3 sec | 30-60 sec |
| RTX 4090 | 1-2 sec | 15-30 sec |
| RTX 5090 | <1 sec | 10-20 sec |

## 💡 Tips

1. Start with Q4_K_M models
2. Use batch workflows
3. Monitor GPU memory
4. Save workflows to `/user/default/workflows/`
5. Use FileBrowser for uploads

---

**Version**: Light (CUDA 13)  
**Size**: ~9GB  
**Startup**: 5-10 minutes (first time)
