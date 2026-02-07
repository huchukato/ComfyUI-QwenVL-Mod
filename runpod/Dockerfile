# ComfyUI-QwenVL-Mod RunPod Worker
# Enhanced with WAN 2.2 workflows, multilingual support, and GGUF backend
# Optimized for RTX 5090 and professional video generation

FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install system dependencies (including CUDA toolkit for TensorRT)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    wget \
    curl \
    unzip \
    pkg-config \
    ca-certificates \
    build-essential \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit for TensorRT support
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

# Configure SSH for root login
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    mkdir -p /run/sshd && \
    rm -f /etc/ssh/ssh_host_*

# Set Python 3.12 as default system-wide
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# Install Miniconda for robust Python environment management
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh && \
    /opt/miniconda3/bin/conda clean -afy

# Create conda environment for ComfyUI with Python 3.12.12
RUN /opt/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/miniconda3/bin/conda create -n comfyui python=3.12.12 -y && \
    /opt/miniconda3/bin/conda clean -afy

ENV PATH="/opt/miniconda3/envs/comfyui/bin:$PATH"

# Install PyTorch with CUDA 13.0 - optimized for RTX 5090
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install ComfyUI 0.12.3 (latest stable)
WORKDIR /ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    git checkout v0.12.3 && \
    pip install -r requirements.txt && \
    pip install GitPython opencv-python

# Install llama-cpp-python for GGUF support (CUDA 13.0 optimized)
RUN pip install --upgrade --force-reinstall --no-cache-dir https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.23-cu130-Basic-linux-20260129/llama_cpp_python-0.3.23+cu130.basic-cp312-cp312-linux_x86_64.whl

# Install all custom nodes from provisioning.sh
WORKDIR /ComfyUI/custom_nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    git clone https://github.com/MadiatorLabs/ComfyUI-RunpodDirect.git && \
    git clone https://github.com/MoonGoblinDev/Civicomfy.git && \
    git clone https://github.com/huchukato/comfy-tagcomplete.git && \
    git clone https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler.git && \
    git clone https://github.com/huchukato/ComfyUI-QwenVL-Mod.git && \
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

# Install custom node dependencies automatically
RUN for node_dir in */; do \
        if [ -f "$node_dir/requirements.txt" ]; then \
            echo "Installing requirements for $node_dir"; \
            pip install --no-cache-dir -r "$node_dir/requirements.txt" || true; \
        fi; \
    done

# Install additional essential dependencies
RUN pip install sageattention torchsde triton opencv-python-headless scikit-image spandrel safetensors jupyterlab

# Install FileBrowser for file management
RUN wget -O /tmp/filebrowser.tar.gz https://github.com/filebrowser/filebrowser/releases/download/v2.28.0/linux-amd64-filebrowser.tar.gz && \
    tar -xzf /tmp/filebrowser.tar.gz -C /tmp && \
    mv /tmp/filebrowser /usr/local/bin/ && \
    chmod +x /usr/local/bin/filebrowser && \
    rm /tmp/filebrowser.tar.gz

# Create FileBrowser database and directories
RUN mkdir -p /filebrowser && \
    mkdir -p /filebrowser/database && \
    mkdir -p /filebrowser/files && \
    ln -sf /ComfyUI /filebrowser/files/ComfyUI && \
    echo '{"port": 8080, "baseURL": "", "address": "0.0.0.0", "log": "stdout", "database": "/filebrowser/database/filebrowser.db", "root": "/filebrowser/files", "noauth": true}' > /filebrowser/.config.json

# Create FileBrowser startup script
RUN echo '#!/bin/bash\n\
echo "📁 Starting FileBrowser on port 8080"\n\
echo "🌐 Access at: http://0.0.0.0:8080"\n\
echo "👤 No authentication required"\n\
echo ""\n\
cd /filebrowser\n\
filebrowser --config /filebrowser/.config.json\n\
' > /usr/local/bin/start-filebrowser.sh && chmod +x /usr/local/bin/start-filebrowser.sh

# Create JupyterLab startup script
RUN echo '#!/bin/bash\n\
echo "📓 Starting JupyterLab on port 8888"\n\
echo "🌐 Access at: http://0.0.0.0:8888"\n\
echo "🔑 No token required"\n\
echo ""\n\
cd /ComfyUI\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token="" --ServerApp.password=""\n\
' > /usr/local/bin/start-jupyter.sh && chmod +x /usr/local/bin/start-jupyter.sh

# Create user directory structure and download workflows
RUN mkdir -p /ComfyUI/user/default/workflows && \
    wget -O /ComfyUI/user/default/workflows/SDXL-LoRaStack-Upscale.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/SDXL-LoRaStack-Upscale.json && \
    wget -O /ComfyUI/user/default/workflows/WAN2.2-I2V-AutoPrompt-1-4.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-AutoPrompt-1-4.json && \
    wget -O /ComfyUI/user/default/workflows/WAN2.2-I2V-AutoPrompt-GGUF-1-4.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-AutoPrompt-GGUF-1-4.json && \
    wget -O /ComfyUI/user/default/workflows/WAN2.2-T2V-AutoPrompt-1-5.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-AutoPrompt-1-5.json && \
    wget -O /ComfyUI/user/default/workflows/WAN2.2-T2V-AutoPrompt-GGUF-1-4.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-AutoPrompt-GGUF-1-4.json && \
    wget -O /ComfyUI/user/default/workflows/Wan2.2-I2V-SVI-AutoPrompt-1-3.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/Wan2.2-I2V-SVI-AutoPrompt-1-3.json && \
    wget -O /ComfyUI/user/default/workflows/Wan2.2-I2V-SVI-AutoPrompt-GGUF-1-1.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/Wan2.2-I2V-SVI-AutoPrompt-GGUF-1-1.json && \
    wget -O /ComfyUI/user/default/workflows/WAN2.2-I2V-Full-AutoPrompt-MMAudio-v1-8.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-Full-AutoPrompt-MMAudio-v1-8.json && \
    wget -O /ComfyUI/user/default/workflows/WAN2.2-I2V-Full-AutoPrompt-MMAudio-GGUF-v1-5.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-Full-AutoPrompt-MMAudio-GGUF-v1-5.json && \
    wget -O /ComfyUI/user/default/workflows/WAN2.2-T2V-Full-AutoPrompt-MMAudio-GGUF.json \
    https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-Full-AutoPrompt-MMAudio-GGUF.json

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
ENV COMFYUI_ARGS="--disable-auto-launch --listen 0.0.0.0 --port 8188 --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload --supports-fp8-compute --fast cublas_ops autotune"

# Create startup script (RunPod style with dynamic workflow updates)
RUN echo '#!/bin/bash\n\
set -e  # Exit the script if any statement returns a non-true return value\n\
\n\
COMFYUI_DIR="/ComfyUI"\n\
FILEBROWSER_CONFIG="/filebrowser/.config.json"\n\
DB_FILE="/filebrowser/database/filebrowser.db"\n\
\n\
# Function to download latest workflows\n\
update_workflows() {\n\
    echo "🔄 Updating workflows from latest versions..."\n\
    cd /ComfyUI/user/default/workflows\n\
    \n\
    # List of latest workflows\n\
    workflows=(\n\
        "SDXL-LoRaStack-Upscale.json"\n\
        "WAN2.2-I2V-AutoPrompt-1-4.json"\n\
        "WAN2.2-I2V-AutoPrompt-GGUF-1-4.json"\n\
        "WAN2.2-T2V-AutoPrompt-1-5.json"\n\
        "WAN2.2-T2V-AutoPrompt-GGUF-1-4.json"\n\
        "Wan2.2-I2V-SVI-AutoPrompt-1-3.json"\n\
        "Wan2.2-I2V-SVI-AutoPrompt-GGUF-1-1.json"\n\
        "WAN2.2-I2V-Full-AutoPrompt-MMAudio-v1-8.json"\n\
        "WAN2.2-I2V-Full-AutoPrompt-MMAudio-GGUF-v1-5.json"\n\
        "WAN2.2-T2V-Full-AutoPrompt-MMAudio-GGUF.json"\n\
    )\n\
    \n\
    for workflow in "${workflows[@]}"; do\n\
        echo "  📥 Downloading: $workflow"\n\
        wget -q "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/$workflow" -O "$workflow" || echo "  ⚠️ Failed to download $workflow"\n\
    done\n\
    \n\
    echo "✅ Workflow update completed"\n\
}\n\
\n\
echo "�🚀 Starting ComfyUI-QwenVL-Mod on RunPod (Full Version)"\n\
echo "🎬 WAN 2.2 workflows ready"\n\
echo "🌐 Multilingual support enabled"\n\
echo "🎨 Visual style detection ready"\n\
echo "⚡ GGUF backend optimized"\n\
echo "🔧 RTX 5090 optimizations active"\n\
echo ""\n\
\n\
# Update workflows to latest versions\n\
update_workflows\n\
\n\
echo "🔧 Services starting:"\n\
echo "📁 FileBrowser: http://0.0.0.0:8080"\n\
echo "📓 JupyterLab:  http://0.0.0.0:8888"\n\
echo "🎨 ComfyUI:    http://0.0.0.0:8188"\n\
echo "📋 Available workflows:"\n\
ls -la /ComfyUI/user/default/workflows/\n\
echo ""\n\
echo "💡 Models pre-loaded for immediate use"\n\
echo ""\n\
\n\
# Start FileBrowser\n\
echo "🔧 Starting FileBrowser..."\n\
nohup /usr/local/bin/start-filebrowser.sh &> /filebrowser.log &\n\
echo "✅ FileBrowser started"\n\
\n\
# Start JupyterLab\n\
echo "🔧 Starting JupyterLab..."\n\
nohup /usr/local/bin/start-jupyter.sh &> /jupyter.log &\n\
echo "✅ JupyterLab started"\n\
\n\
# Start ComfyUI\n\
echo "🚀 Starting ComfyUI..."\n\
echo "🌐 ComfyUI will be available at: http://0.0.0.0:8188"\n\
cd $COMFYUI_DIR\n\
nohup /opt/venv/bin/python main.py --disable-auto-launch --listen 0.0.0.0 --port 8188 --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload --supports-fp8-compute --fast cublas_ops autotune &> /comfyui.log &\n\
echo "✅ ComfyUI started"\n\
\n\
# Tail the ComfyUI log\n\
tail -f /comfyui.log\n\
' > /ComfyUI/start.sh && chmod +x /ComfyUI/start.sh

# Set working directory
WORKDIR /ComfyUI

# Expose ports
EXPOSE 8080 8188 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://0.0.0.0:8188/system_stats || exit 1

# Start ComfyUI
CMD ["/ComfyUI/start.sh"]
