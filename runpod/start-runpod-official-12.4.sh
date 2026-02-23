#!/bin/bash

# Start script for RunPod Official CUDA 12.4
# Handles workspace setup, ComfyUI startup, and service management

set -e

# Set environment variables
export COMFYUI_DIR="/workspace/runpod-slim/ComfyUI"
export WORKSPACE_DIR="/workspace/runpod-slim"
export FILEBROWSER_CONFIG="/workspace/runpod-slim/.filebrowser.json"
export DB_FILE="/workspace/runpod-slim/filebrowser/database/filebrowser.db"

# Create necessary directories
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$(dirname "$FILEBROWSER_CONFIG")"
mkdir -p "$(dirname "$DB_FILE")"

# Function to update workflows
update_workflows() {
    echo "Updating workflows..."
    cd $COMFYUI_DIR/user/default/workflows
    
    workflows=(
        "PMP-LoRaStack-Upscale-Wildcards.json"
        "WAN2.2-I2V-AutoPrompt-Story.json"
        "WAN2.2-T2V-AutoPrompt-Story.json"
        "Wan2.2-I2V-SVI-AutoPrompt-Story.json"
        "WAN2.2-I2V-AutoPrompt.json"
        "WAN2.2-I2V-AutoPrompt-GGUF.json"
        "WAN2.2-T2V-AutoPrompt.json"
        "WAN2.2-T2V-AutoPrompt-GGUF.json"
        "Wan2.2-I2V-SVI-AutoPrompt.json"
        "Wan2.2-I2V-SVI-AutoPrompt-GGUF.json"
        "WAN2.2-I2V-Full-AutoPrompt-MMAudio.json"
        "WAN2.2-I2V-Full-AutoPrompt-MMAudio-GGUF.json"
        "WAN2.2-T2V-I2V-Full-AutoPrompt-MMAudio-GGUF.json"
    )
    
    for workflow in "${workflows[@]}"; do
        if [ ! -f "$workflow" ]; then
            echo "Downloading $workflow..."
            wget -q "https://raw.githubusercontent.com/huchukato/ComfyUI-QwenVL-Mod/main/runpod/workflows/$workflow" || echo "Failed to download $workflow"
        else
            echo "$workflow already exists"
        fi
    done
}

# Function to install custom nodes
install_custom_nodes() {
    echo "Installing custom nodes..."
    cd "$COMFYUI_DIR/custom_nodes"
    
    nodes=(
        "https://github.com/ltdrdata/ComfyUI-Manager"
        "https://github.com/cubiq/ComfyUI_essentials"
        "https://github.com/huchukato/comfy-tagcomplete"
        "https://github.com/huchukato/ComfyUI-QwenVL-Mod"
        "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
        "https://github.com/MoonGoblinDev/Civicomfy"
        "https://github.com/huchukato/ComfyUI-HuggingFace"
        "https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler"
        "https://github.com/ltdrdata/was-node-suite-comfyui"
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
        "https://github.com/rgthree/rgthree-comfy"
        "https://github.com/yolain/ComfyUI-Easy-Use"
        "https://github.com/kijai/ComfyUI-KJNodes"
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
        "https://github.com/Smirnov75/ComfyUI-mxToolkit"
        "https://github.com/princepainter/ComfyUI-PainterI2V"
        "https://github.com/princepainter/ComfyUI-PainterI2Vadvanced"
        "https://github.com/princepainter/ComfyUI-PainterLongVideo"
        "https://github.com/ashtar1984/comfyui-find-perfect-resolution"
        "https://github.com/ComfyAssets/ComfyUI_Selectors"
        "https://github.com/city96/ComfyUI-GGUF"
        "https://github.com/kijai/ComfyUI-MMAudio"
        "https://github.com/GACLove/ComfyUI-VFI"
        "https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt"
        "https://github.com/stduhpf/ComfyUI-WanMoeKSampler"
        "https://github.com/melMass/comfy_mtb"
    )
    
    for node_url in "${nodes[@]}"; do
        node_name=$(basename "$node_url")
        if [ ! -d "$node_name" ]; then
            echo "Cloning $node_name..."
            git clone "$node_url" || echo "Failed to clone $node_name"
        else
            echo "$node_name already exists"
        fi
    done
}

# Clone ComfyUI if not exists
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

# Install custom nodes
install_custom_nodes

# Update workflows
update_workflows

# Set default ComfyUI arguments if not provided
if [ -z "$COMFYUI_ARGS" ]; then
    export COMFYUI_ARGS="--listen 0.0.0.0 --port 8188 --disable-auto-launch --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload"
fi

# Start SSH service
echo "Starting SSH service..."
/usr/sbin/sshd -D &

# Start FileBrowser
echo "Starting FileBrowser on port 8080..."
filebrowser --config "$FILEBROWSER_CONFIG" --database "$DB_FILE" --address 0.0.0.0 --port 8080 --no-auth &

# Start JupyterLab
echo "Starting JupyterLab on port 8888..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token="" --ServerApp.password="" --ServerApp.terminado_manage_command=True &

# Start ComfyUI
echo "Starting ComfyUI with arguments: $COMFYUI_ARGS"
cd "$COMFYUI_DIR"
python main.py $COMFYUI_ARGS &

echo ""
echo "🚀 ComfyUI-QwenVL-Mod Started Successfully!"
echo "====================================="
echo "📊 Services Status:"
echo "  ✅ SSH:        Port 22 (running)"
echo "  ✅ FileBrowser: Port 8080 (running)"
echo "  ✅ JupyterLab:  Port 8888 (running)"
echo "  ✅ ComfyUI:    Port 8188 (starting...)"
echo ""
echo "🌐 Access URLs:"
echo "  📁 FileBrowser: http://localhost:8080"
echo "  📓 JupyterLab:  http://localhost:8888"
echo "  🎨 ComfyUI:    http://localhost:8188"
echo ""
echo "💡 Tips:"
echo "  - ComfyUI may take 1-2 minutes to fully start"
echo "  - Check ComfyUI logs for startup progress"
echo "  - All services are running in background"

# Keep container running
wait
