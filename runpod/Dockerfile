# ComfyUI-QwenVL-Mod RunPod Worker
# Enhanced with WAN 2.2 workflows, multilingual support, and GGUF backend
# Optimized for RTX 5090 and professional video generation

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI
WORKDIR /ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install -r requirements.txt

# Install llama-cpp-python for GGUF support (CUDA 12.4 optimized)
RUN pip install --upgrade --force-reinstall llama-cpp-python --prefer-binary --extra-index-url=https://download.pytorch.org/whl/cu124

# Install ComfyUI-QwenVL-Mod
WORKDIR /ComfyUI/custom_nodes
RUN git clone https://github.com/huchukato/ComfyUI-QwenVL-Mod.git

# Install QwenVL-Mod dependencies
RUN cd ComfyUI-QwenVL-Mod && pip install -r requirements.txt

# Install essential custom nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    git clone https://github.com/MoonGoblinDev/Civicomfy.git && \
    git clone https://github.com/huchukato/comfy-tagcomplete.git && \
    git clone https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler.git && \
    git clone https://github.com/ltdrdata/was-node-suite-comfyui.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git && \
    git clone https://github.com/princepainter/ComfyUI-PainterI2V.git && \
    git clone https://github.com/princepainter/ComfyUI-PainterI2Vadvanced.git && \
    git clone https://github.com/princepainter/ComfyUI-PainterLongVideo.git && \
    git clone https://github.com/ashtar1984/comfyui-find-perfect-resolution.git && \
    git clone https://github.com/ComfyAssets/ComfyUI_Selectors.git && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    git clone https://github.com/kijai/ComfyUI-MMAudio.git && \
    git clone https://github.com/GACLove/ComfyUI-VFI.git && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    git clone https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt.git && \
    git clone https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto.git && \
    git clone https://github.com/stduhpf/ComfyUI-WanMoeKSampler.git && \
    git clone https://github.com/melMass/comfy_mtb.git

# Install node dependencies
RUN cd ComfyUI-Manager && pip install -r requirements.txt && \
    cd ../ComfyUI_essentials && pip install -r requirements.txt && \
    cd ../ComfyUI-Impact-Pack && pip install -r requirements.txt && \
    cd ../ComfyUI-VideoHelperSuite && pip install -r requirements.txt && \
    cd ../ComfyUI-GGUF && pip install -r requirements.txt && \
    cd ../ComfyUI-MMAudio && pip install -r requirements.txt && \
    cd ../ComfyUI-VFI && pip install -r requirements.txt && \
    cd ../ComfyUI-RIFE-TensorRT-Auto && pip install -r requirements.txt

# Create workflows directory
RUN mkdir -p /ComfyUI/workflows

# Download essential WAN 2.2 workflows
RUN wget -O /ComfyUI/workflows/WAN2.2-I2V-AutoPrompt.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-AutoPrompt-1-4.json && \
    wget -O /ComfyUI/workflows/WAN2.2-T2V-AutoPrompt.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-AutoPrompt-1-4.json && \
    wget -O /ComfyUI/workflows/WAN2.2-I2V-SVI-AutoPrompt.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/Wan2.2-I2V-SVI-AutoPrompt-1-2.json && \
    wget -O /ComfyUI/workflows/SDXL-LoRaStack-Upscale.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/SDXL-LoRaStack-Upscale.json

# Create models directories
RUN mkdir -p /ComfyUI/models/vae && \
    mkdir -p /ComfyUI/models/upscale_models && \
    mkdir -p /ComfyUI/models/text_encoders

# Download essential models
RUN wget -O /ComfyUI/models/vae/wan_2.1_vae.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors && \
    wget -O /ComfyUI/models/vae/sdxl_vae.safetensors \
    https://huggingface.co/huchukato/favs/resolve/main/VAE/sdxl.vae.safetensors && \
    wget -O /ComfyUI/models/upscale_models/2xLexicaRRDBNet.pth \
    https://huggingface.co/huchukato/favs/resolve/main/ESRGAN/2xLexicaRRDBNet.pth && \
    wget -O /ComfyUI/models/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors \
    https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors

# Set ComfyUI arguments
ENV COMFYUI_ARGS="--disable-auto-launch --port 8080 --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload --supports-fp8-compute --fast cublas_ops autotune"

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting ComfyUI-QwenVL-Mod on RunPod"\n\
echo "🎬 WAN 2.2 workflows loaded"\n\
echo "🌐 Multilingual support enabled"\n\
echo "🎨 Visual style detection ready"\n\
echo "⚡ GGUF backend optimized"\n\
echo "🔧 RTX 5090 optimizations active"\n\
echo ""\n\
echo "📋 Available workflows:"\n\
ls -la /ComfyUI/workflows/\n\
echo ""\n\
echo "🌐 ComfyUI will be available at: http://localhost:8080"\n\
echo ""\n\
exec python main.py $COMFYUI_ARGS\n\
' > /ComfyUI/start.sh && chmod +x /ComfyUI/start.sh

# Set working directory
WORKDIR /ComfyUI

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/system_stats || exit 1

# Start ComfyUI
CMD ["/ComfyUI/start.sh"]
