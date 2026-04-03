# ComfyUI-QwenVL-Mod RunPod Worker
# Enhanced with WAN 2.2 workflows, multilingual support, and GGUF backend
# Optimized for RTX 5090 and professional video generation

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Install system dependencies (CUDA already included in base image)
RUN apt update && \
    apt install -y --no-install-recommends \
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
    openssh-client \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Python version
RUN python3.12 --version

# CUDA toolkit and cuDNN already included in nvidia/cuda base image

# Configure SSH for root login
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    mkdir -p /run/sshd && \
    rm -f /etc/ssh/ssh_host_*

# Create virtual environment to avoid externally-managed-environment error
RUN python3.12 -m venv /opt/comfyui-env && \
    /opt/comfyui-env/bin/pip install --upgrade pip

# Install PyTorch with CUDA 12.8 (compatible with base image CUDA 12.8.0) - optimized for RTX 5090
RUN /opt/comfyui-env/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    /opt/comfyui-env/bin/pip cache purge

# Remove uv to force ComfyUI-Manager to use pip (uv doesn't respect --system-site-packages properly)
RUN pip uninstall -y uv 2>/dev/null || true && \
    rm -f /usr/local/bin/uv /usr/local/bin/uvx

# Install ComfyUI latest stable version
WORKDIR /ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    /opt/comfyui-env/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/comfyui-env/bin/pip install --no-cache-dir GitPython opencv-python && \
    /opt/comfyui-env/bin/pip cache purge

# Install llama-cpp-python for GGUF support (CUDA 12.8 compatible with base image)
RUN /opt/comfyui-env/bin/pip install --upgrade --force-reinstall --no-cache-dir \
    https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.34-cu128-Basic-linux-20260331/llama_cpp_python-0.3.34+cu128.basic-cp312-cp312-linux_x86_64.whl && \
    /opt/comfyui-env/bin/pip cache purge

# Install ALL ComfyUI custom nodes from VastAI provisioning + RunpodDirect
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/kijai/ComfyUI_essentials.git && \
    git clone https://github.com/huchukato/comfy-tagcomplete.git && \
    git clone https://github.com/huchukato/ComfyUI-QwenVL-Mod.git && \
    git clone https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto.git && \
    git clone https://github.com/huchukato/ComfyUI-Upscaler-TensorRT-Auto.git && \
    git clone https://github.com/huchukato/ComfyUI-HuggingFace.git && \
    git clone https://github.com/MadiatorLabs/ComfyUI-RunpodDirect.git && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    git clone https://github.com/MoonGoblinDev/Civicomfy.git && \
    git clone https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler.git && \
    git clone https://github.com/ltdrdata/was-node-suite-comfyui.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git && \
    git clone https://github.com/princepainter/ComfyUI-PainterI2V.git && \
    git clone https://github.com/princepainter/ComfyUI-PainterLongVideo.git && \
    git clone https://github.com/ashtar1984/comfyui-find-perfect-resolution.git && \
    git clone https://github.com/huchukato/ComfyUI-Selectors.git && \
    git clone https://github.com/kijai/ComfyUI-MMAudio.git && \
    git clone https://github.com/GACLove/ComfyUI-VFI.git && \
    git clone https://github.com/stduhpf/ComfyUI-WanMoeKSampler.git && \
    git clone https://github.com/melMass/comfy_mtb.git

# Copy wildcards to Easy-Use (required for wildcard workflows)
RUN cp -r /ComfyUI/custom_nodes/comfy-tagcomplete/wildcards/mbe /ComfyUI/custom_nodes/ComfyUI-Easy-Use/wildcards || echo "Wildcards copy completed (or already exists)"

# Install requirements for all custom nodes
RUN bash -c 'cd /ComfyUI/custom_nodes && for node_dir in */; do if [ -f "$node_dir/requirements.txt" ]; then echo "Installing requirements for $node_dir..." && /opt/comfyui-env/bin/pip install --no-cache-dir -r "$node_dir/requirements.txt" || echo "Failed to install requirements for $node_dir"; fi; done' && \
    /opt/comfyui-env/bin/pip cache purge

# Install additional essential dependencies
RUN /opt/comfyui-env/bin/pip install --no-cache-dir jupyter sageattention && \
    /opt/comfyui-env/bin/pip cache purge

# Clean up heavy Git files to save space (keep repo structure for updates)
RUN find /ComfyUI/custom_nodes -name ".git" -type d -exec find {} -name "*.pack" -delete + || true && \
    find /ComfyUI/custom_nodes -name ".git" -type d -exec find {} -name "pack-*.idx" -delete + || true && \
    find /ComfyUI/custom_nodes -name ".git" -type d -exec find {} -name "objects" -type d -exec rm -rf {} + || true

# Create FileBrowser (RunPod method)
RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

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

# Create Jupyter Lab startup script
RUN cat > /usr/local/bin/start-jupyter.sh << 'EOF'
#!/bin/bash
echo "📓 Starting Jupyter Lab on port 8888"
echo "🌐 Access at: http://0.0.0.0:8888"
echo "🔑 No token required"
echo ""
cd /ComfyUI
echo "🔧 Starting Jupyter Lab with terminal support..."
nohup /opt/comfyui-env/bin/jupyter lab \
    --allow-root \
    --no-browser \
    --port=8888 \
    --ip=0.0.0.0 \
    --FileContentsManager.delete_to_trash=False \
    --FileContentsManager.preferred_dir=/ComfyUI \
    --ServerApp.root_dir=/ComfyUI \
    --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
    --IdentityProvider.token="" \
    --ServerApp.allow_origin=* &> /jupyter.log &
echo "✅ Jupyter Lab started"
echo "🔍 Checking Jupyter Lab status..."
sleep 5
if pgrep -f "jupyter lab" > /dev/null; then
    echo "✅ Jupyter Lab started successfully"
    echo "📊 Jupyter Lab logs:"
    tail -10 /jupyter.log
else
    echo "❌ Jupyter Lab failed to start"
    echo "📊 Full error logs:"
    cat /jupyter.log
    echo "🔍 Checking what went wrong..."
    echo "🐍 Python version in virtual env:"
    /opt/comfyui-env/bin/python --version
    echo "📦 Jupyter Lab version in virtual env:"
    /opt/comfyui-env/bin/jupyter lab --version
fi
EOF
RUN chmod +x /usr/local/bin/start-jupyter.sh

# Create user directory structure
RUN mkdir -p /ComfyUI/user/default/workflows

# Create models directories
RUN mkdir -p /ComfyUI/models/vae && \
    mkdir -p /ComfyUI/models/upscale_models && \
    mkdir -p /ComfyUI/models/text_encoders

# Clean up heavy Git files to save space (keep repo structure for updates)
RUN find /ComfyUI/custom_nodes -name ".git" -type d -exec find {} -name "*.pack" -delete + || true && \
    find /ComfyUI/custom_nodes -name ".git" -type d -exec find {} -name "pack-*.idx" -delete + || true && \
    find /ComfyUI/custom_nodes -name ".git" -type d -exec find {} -name "objects" -type d -exec rm -rf {} + || true

# Aggressive cleanup to reduce image size
RUN rm -rf /tmp/* \
       && rm -rf /var/lib/apt/lists/* \
       && rm -rf /var/cache/apt/archives/* \
       && find /ComfyUI -name "*.pyc" -delete \
       && find /ComfyUI -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
       && /opt/comfyui-env/bin/pip cache purge

# Set ComfyUI arguments (optimized for WAN2.2 with SageAttention)
ENV COMFYUI_ARGS="--listen 0.0.0.0 --port 8188 --disable-auto-launch --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload"

# Create startup script (RunPod style with dynamic workflow updates)
RUN cat > /ComfyUI/start.sh << 'EOF'
#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

COMFYUI_DIR="/ComfyUI"
FILEBROWSER_CONFIG="/filebrowser/.config.json"
DB_FILE="/filebrowser/database/filebrowser.db"

# Function to update custom nodes
update_nodes() {
    echo "🔄 Updating custom nodes..."
    cd /ComfyUI/custom_nodes
    
    # List of custom nodes to update
    nodes=(
        "https://github.com/ltdrdata/ComfyUI-Manager.git"
        "https://github.com/kijai/ComfyUI_essentials.git"
        "https://github.com/huchukato/comfy-tagcomplete.git"
        "https://github.com/huchukato/ComfyUI-QwenVL-Mod.git"
        "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto.git"
        "https://github.com/huchukato/ComfyUI-Upscaler-TensorRT-Auto.git"
        "https://github.com/huchukato/ComfyUI-HuggingFace.git"
        "https://github.com/MadiatorLabs/ComfyUI-RunpodDirect.git"
        "https://github.com/city96/ComfyUI-GGUF.git"
        "https://github.com/MoonGoblinDev/Civicomfy.git"
        "https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler.git"
        "https://github.com/ltdrdata/was-node-suite-comfyui.git"
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
        "https://github.com/rgthree/rgthree-comfy.git"
        "https://github.com/yolain/ComfyUI-Easy-Use.git"
        "https://github.com/kijai/ComfyUI-KJNodes.git"
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"
        "https://github.com/Smirnov75/ComfyUI-mxToolkit.git"
        "https://github.com/princepainter/ComfyUI-PainterI2V.git"
        "https://github.com/princepainter/ComfyUI-PainterLongVideo.git"
        "https://github.com/ashtar1984/comfyui-find-perfect-resolution.git"
        "https://github.com/huchukato/ComfyUI-Selectors.git"
        "https://github.com/kijai/ComfyUI-MMAudio.git"
        "https://github.com/GACLove/ComfyUI-VFI.git"
        "https://github.com/stduhpf/ComfyUI-WanMoeKSampler.git"
        "https://github.com/melMass/comfy_mtb.git"
    )
    
    for node_url in "${nodes[@]}"; do
        node_name=$(basename "$node_url" .git)
        if [ -d "$node_name" ]; then
            echo "  🔄 Updating: $node_name"
            cd "$node_name"
            git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || echo "  ⚠️ Failed to update $node_name"
            cd ..
        else
            echo "  📥 Installing: $node_name"
            git clone "$node_url" || echo "  ⚠️ Failed to clone $node_name"
        fi
    done
    
    # Install requirements for updated nodes
    echo "📦 Installing updated requirements..."
    bash -c 'for node_dir in */; do if [ -f "$node_dir/requirements.txt" ]; then echo "Installing requirements for $node_dir..." && /opt/comfyui-env/bin/pip install --no-cache-dir -r "$node_dir/requirements.txt" || echo "Failed to install requirements for $node_dir"; fi; done'
    
    echo "✅ Node update completed"
}

# Function to download latest workflows
update_workflows() {
    echo "🔄 Updating workflows from latest versions..."
    mkdir -p /ComfyUI/user/default/workflows
    cd /ComfyUI/user/default/workflows
    
    # List of latest workflows
    workflows=(
        "PMP-LoRaStack-Upscale-Wildcards.json"
        "WAN2.2-I2V-AutoPrompt-Story.json"
        "WAN2.2-T2V-I2V-AutoPrompt-Story.json"
        "WAN2.2-I2V-SVI-AutoPrompt-Story.json"
        "WAN2.2-I2V-AutoPrompt.json"
        "WAN2.2-I2V-AutoPrompt-GGUF.json"
        "WAN2.2-T2V-AutoPrompt.json"
        "WAN2.2-T2V-AutoPrompt-GGUF.json"
        "WAN2.2-I2V-SVI-AutoPrompt.json"
        "WAN2.2-I2V-SVI-AutoPrompt-GGUF.json"
        "WAN2.2-I2V-Full-AutoPrompt-MMAudio.json"
        "WAN2.2-I2V-Full-AutoPrompt-MMAudio-GGUF.json"
        "WAN2.2-T2V-I2V-Full-AutoPrompt-MMAudio-GGUF.json"
    )
    
    for workflow in "${workflows[@]}"; do
        echo "  📥 Downloading: $workflow"
        wget -q "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/$workflow" -O "$workflow" || echo "  ⚠️ Failed to download $workflow"
    done
    
    echo "✅ Workflow update completed"
}

download_models() {
    echo "🔄 Downloading essential models..."
    cd /ComfyUI
    
    # Download VAE models
    if [ ! -f "/ComfyUI/models/vae/wan_2.1_vae.safetensors" ]; then
        echo "  📥 Downloading WAN 2.1 VAE..."
        wget -q "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" -O "/ComfyUI/models/vae/wan_2.1_vae.safetensors" || echo "  ⚠️ Failed to download WAN 2.1 VAE"
    fi
    
        
    if [ ! -f "/ComfyUI/models/vae/sdxl_vae.safetensors" ]; then
        echo "  📥 Downloading SDXL VAE..."
        wget -q "https://huggingface.co/huchukato/favs/resolve/main/VAE/sdxl.vae.safetensors" -O "/ComfyUI/models/vae/sdxl_vae.safetensors" || echo "  ⚠️ Failed to download SDXL VAE"
    fi
    
    # Download upscale models
    if [ ! -f "/ComfyUI/models/upscale_models/2xLexicaRRDBNet.pth" ]; then
        echo "  📥 Downloading 2xLexicaRRDBNet upscale model..."
        wget -q "https://huggingface.co/huchukato/favs/resolve/main/ESRGAN/2xLexicaRRDBNet.pth" -O "/ComfyUI/models/upscale_models/2xLexicaRRDBNet.pth" || echo "  ⚠️ Failed to download upscale model"
    fi
    
    if [ ! -f "/ComfyUI/models/upscale_models/2xLexicaRRDBNet_Sharp.pth" ]; then
        echo "  📥 Downloading 2xLexicaRRDBNet Sharp upscale model..."
        wget -q "https://huggingface.co/huchukato/favs/resolve/main/ESRGAN/2xLexicaRRDBNet_Sharp.pth" -O "/ComfyUI/models/upscale_models/2xLexicaRRDBNet_Sharp.pth" || echo "  ⚠️ Failed to download Sharp upscale model"
    fi
    
    # Download text encoder
    if [ ! -f "/ComfyUI/models/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" ]; then
        echo "  📥 Downloading NSFW WAN UMT5-XXL text encoder..."
        wget -q "https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" -O "/ComfyUI/models/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" || echo "  ⚠️ Failed to download text encoder"
    fi
    
        
    echo "✅ Model download completed"
    echo "📋 Available models:"
    ls -la /ComfyUI/models/vae/
    ls -la /ComfyUI/models/upscale_models/
    ls -la /ComfyUI/models/text_encoders/
}

echo "🚀 Starting ComfyUI-QwenVL-Mod on RunPod (Full Version)"
echo "🎬 WAN 2.2 workflows ready"
echo "🌐 Multilingual support enabled"
echo "🎨 Visual style detection ready"
echo "⚡ GGUF backend optimized"
echo ""

# Update workflows to latest versions
update_workflows

# Update custom nodes to latest versions
update_nodes

# Download essential models
download_models

echo "🔧 Services starting:"
echo "📁 FileBrowser: http://0.0.0.0:8080"
echo "📓 Jupyter Lab:  http://0.0.0.0:8888 (Terminal included)"
echo "🎨 ComfyUI:    http://0.0.0.0:8188"
echo "📋 Available workflows:"
ls -la /ComfyUI/user/default/workflows/
echo ""
echo "💡 Models pre-loaded for immediate use"
echo ""

# Start FileBrowser
echo "🔧 Starting FileBrowser..."
nohup /usr/local/bin/start-filebrowser.sh &> /filebrowser.log &
echo "✅ FileBrowser started"

# Start Jupyter Lab
echo "🔧 Starting Jupyter Lab..."
nohup /usr/local/bin/start-jupyter.sh &> /jupyter.log &
sleep 5
echo "🔍 Checking Jupyter Lab status..."
if pgrep -f "jupyter lab" > /dev/null; then
    echo "✅ Jupyter Lab started successfully"
    echo "📊 Jupyter Lab logs:"
    tail -10 /jupyter.log
else
    echo "❌ Jupyter Lab failed to start"
    echo "📊 Full error logs:"
    cat /jupyter.log
    echo "🔍 Checking what went wrong..."
    echo "🐍 Python version in virtual env:"
    /opt/comfyui-env/bin/python --version
    echo "📦 Jupyter Lab version in virtual env:"
    /opt/comfyui-env/bin/jupyter lab --version
fi


# Start ComfyUI
echo "🚀 Starting ComfyUI..."
echo "🌐 ComfyUI will be available at: http://0.0.0.0:$(echo $COMFYUI_ARGS | grep -oP '(?<=--port\s)\d+' || echo '8188')"
echo "📓 Jupyter Lab Terminal: Open http://0.0.0.0:8888 → New → Terminal"
cd $COMFYUI_DIR
nohup /opt/comfyui-env/bin/python main.py $COMFYUI_ARGS &> /comfyui.log &
echo "✅ ComfyUI started"

# Tail the ComfyUI log
tail -f /comfyui.log
EOF
RUN chmod +x /ComfyUI/start.sh

# Set working directory
WORKDIR /ComfyUI

# Expose ports
EXPOSE 8080 8188 8888 22

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://0.0.0.0:8188/system_stats || exit 1

# Start ComfyUI
CMD ["/ComfyUI/start.sh"]
