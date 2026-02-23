#!/bin/bash
set -e

COMFYUI_DIR="/workspace/runpod-slim/ComfyUI"
FILEBROWSER_CONFIG="/workspace/runpod-slim/.filebrowser.json"
DB_FILE="/workspace/runpod-slim/filebrowser/database/filebrowser.db"

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

start_jupyter() {
    echo "Starting Jupyter Lab..."
    cd $COMFYUI_DIR
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --ServerApp.token="" \
        --ServerApp.password="" \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
        --IdentityProvider.token="" \
        --ServerApp.allow_origin=* &> /jupyter.log &
}

start_filebrowser() {
    echo "Starting FileBrowser..."
    mkdir -p $(dirname $FILEBROWSER_CONFIG)
    mkdir -p $(dirname $DB_FILE)
    
    if [ ! -f "$FILEBROWSER_CONFIG" ]; then
        cat > $FILEBROWSER_CONFIG << 'FBEOF'
{
  "port": 8080,
  "baseurl": "",
  "log": "stdout",
  "database": "/workspace/runpod-slim/filebrowser/database/filebrowser.db",
  "root": "/workspace/runpod-slim"
}
FBEOF
    fi
    
    filebrowser -c "$FILEBROWSER_CONFIG" &> /filebrowser.log &
}

start_comfyui() {
    echo "Starting ComfyUI..."
    echo "ComfyUI will be available at: http://0.0.0.0:$(echo $COMFYUI_ARGS | grep -oP '(?<=--port\s)\d+' || echo '8188')"
    cd $COMFYUI_DIR
    python main.py $COMFYUI_ARGS &> /comfyui.log &
}

echo "Starting RunPod QwenVL Mod Environment (CUDA 12.4)..."
update_workflows
start_jupyter
start_filebrowser
start_comfyui

echo "Services started:"
echo "ComfyUI: http://0.0.0.0:$(echo $COMFYUI_ARGS | grep -oP '(?<=--port\s)\d+' || echo '8188')"
echo "Jupyter Lab: http://0.0.0.0:8888"
echo "FileBrowser: http://0.0.0.0:8080"
echo "SSH: port 22"

while true; do
    echo "=== Service Status ==="
    if pgrep -f "python main.py" > /dev/null; then
        echo "ComfyUI: Running"
    else
        echo "ComfyUI: Not running"
    fi
    
    if pgrep -f "jupyter lab" > /dev/null; then
        echo "Jupyter Lab: Running"
    else
        echo "Jupyter Lab: Not running"
    fi
    
    if pgrep -f "filebrowser" > /dev/null; then
        echo "FileBrowser: Running"
    else
        echo "FileBrowser: Not running"
    fi
    
    echo "=== Recent ComfyUI Logs ==="
    tail -5 /comfyui.log 2>/dev/null || echo "No ComfyUI logs yet"
    
    sleep 30
done
