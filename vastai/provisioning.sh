#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI

APT_PACKAGES=(
    #"package-1"
    #"package-2"
)

PIP_PACKAGES=(
    "--upgrade --force-reinstall --no-cache-dir https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.27-cu130-Basic-linux-20260222/llama_cpp_python-0.3.27+cu130.basic-cp312-cp312-linux_x86_64.whl"
)

NODES=(
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

WORKFLOWS=(
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/PMP-LoRaStack-Upscale-Wildcards.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-AutoPrompt-Story.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-AutoPrompt-Story.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/Wan2.2-I2V-SVI-AutoPrompt-Story.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-AutoPrompt.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-AutoPrompt-GGUF.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-AutoPrompt.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-AutoPrompt-GGUF.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/Wan2.2-I2V-SVI-AutoPrompt.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/Wan2.2-I2V-SVI-AutoPrompt-GGUF.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-Full-AutoPrompt-MMAudio.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-I2V-Full-AutoPrompt-MMAudio-GGUF.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/WAN2.2-T2V-I2V-Full-AutoPrompt-MMAudio-GGUF.json"
)


CHECKPOINT_MODELS=(
)

UNET_MODELS=(
   
)

LORA_MODELS=(
            
)

VAE_MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors"
    "https://huggingface.co/huchukato/favs/resolve/main/VAE/sdxl.vae.safetensors"
)

ESRGAN_MODELS=(
    "https://huggingface.co/huchukato/favs/resolve/main/ESRGAN/2xLexicaRRDBNet.pth"
    "https://huggingface.co/huchukato/favs/resolve/main/ESRGAN/2xLexicaRRDBNet_Sharp.pth"
)

TEXT_ENCODERS=(
    "https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)

CONTROLNET_MODELS=(
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    provisioning_print_header
    echo "🚀 Starting provisioning process..."
    
    echo "📦 Installing APT packages..."
    provisioning_get_apt_packages
    
    echo "🔧 Installing custom nodes..."
    provisioning_get_nodes
    
    echo "� Copying wildcards to Impact-Pack..."
    provisioning_copy_wildcards
    
    echo "� Installing PIP packages..."
    provisioning_get_pip_packages
    
    echo "📁 Downloading workflows..."
    # Verify ComfyUI directory exists
    if [ ! -d "$COMFYUI_DIR" ]; then
        echo "⚠️ ComfyUI directory not found: $COMFYUI_DIR"
        echo "🔍 Creating ComfyUI directory..."
        mkdir -p "$COMFYUI_DIR"
    fi
    
    # Create workflows directory
    mkdir -p "${COMFYUI_DIR}/user/default/workflows"
    
    provisioning_get_files \
        "${COMFYUI_DIR}/user/default/workflows" \
        "${WORKFLOWS[@]}"
        
    echo "✅ Workflows downloaded to: ${COMFYUI_DIR}/user/default/workflows"
        
    echo "🎯 Downloading checkpoint models..."
    provisioning_get_files \
        "${COMFYUI_DIR}/models/checkpoints" \
        "${CHECKPOINT_MODELS[@]}"
        
    echo "🧠 Downloading U-NET models..."
    provisioning_get_files \
        "${COMFYUI_DIR}/models/unet" \
        "${UNET_MODELS[@]}"
        
    echo "🎨 Downloading LoRA models..."
    provisioning_get_files \
        "${COMFYUI_DIR}/models/lora" \
        "${LORA_MODELS[@]}"
        
    echo "🎮 Downloading ControlNet models..."
    provisioning_get_files \
        "${COMFYUI_DIR}/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
        
    echo "🔮 Downloading VAE models..."
    provisioning_get_files \
        "${COMFYUI_DIR}/models/vae" \
        "${VAE_MODELS[@]}"
        
    echo "⚡ Downloading upscale models..."
    provisioning_get_files \
        "${COMFYUI_DIR}/models/upscale_models" \
        "${ESRGAN_MODELS[@]}"
        
    echo "📝 Downloading text encoders..."
    provisioning_get_files \
        "${COMFYUI_DIR}/models/text_encoders" \
        "${TEXT_ENCODERS[@]}"        
        
    provisioning_print_end
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
            sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
           echo "Installing PIP packages..."
           for package in "${PIP_PACKAGES[@]}"; do
               echo "Installing: $package"
               pip install --root-user-action=ignore --no-cache-dir $package
               echo "✓ Completed: $package"
           done
           echo "All PIP packages installed successfully!"
    fi
}

function provisioning_get_nodes() {
    echo "Processing ${#NODES[@]} nodes..."
    local count=0
    for repo in "${NODES[@]}"; do
        ((count++))
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        
        echo "[$count/${#NODES[@]}] Processing node: $dir"
        
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                echo "  → Updating existing node..."
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                   echo "  → Installing requirements..."
                   pip install --root-user-action=ignore --no-cache-dir -r "$requirements"
                fi
            else
                echo "  → Node exists, skipping (AUTO_UPDATE=false)"
            fi
        else
            echo "  → Downloading new node..."
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                echo "  → Installing requirements..."
                pip install --root-user-action=ignore --no-cache-dir -r "${requirements}"
            fi
        fi
        
        # Special handling for ComfyUI-Easy-Use-Frontend
        if [[ "$dir" == "ComfyUI-Easy-Use" ]]; then
            echo "  → Installing ComfyUI-Easy-Use-Frontend..."
            cd "${path}"
            if [[ ! -d "ComfyUI-Easy-Use-Frontend" ]]; then
                git clone https://github.com/yolain/ComfyUI-Easy-Use-Frontend.git
            fi
            cd ComfyUI-Easy-Use-Frontend
            if [[ ! -d "node_modules" ]]; then
                echo "    → Installing npm dependencies..."
                npm install
            fi
            echo "    → Building frontend..."
            npm run build:dev
            # Create config.yaml with WEB_VERSION: dev
            if [[ ! -f "${path}/config.yaml" ]]; then
                echo "WEB_VERSION: dev" > "${path}/config.yaml"
            fi
            cd "${COMFYUI_DIR}"
        fi
    done
    echo "All nodes processed successfully!"
}

function provisioning_copy_wildcards() {
    echo "Copying wildcards from comfy-tagcomplete to Impact-Pack..."
    
    local source_dir="${COMFYUI_DIR}/custom_nodes/comfy-tagcomplete/wildcards/mbe"
    local target_dir="${COMFYUI_DIR}/custom_nodes/ComfyUI-Impact-Pack/wildcards"
    
    # Create target directory if it doesn't exist
    mkdir -p "$target_dir"
    
    # Copy the entire mbe directory
    if [[ -d "$source_dir" ]]; then
        echo "  → Copying mbe wildcards..."
        cp -r "$source_dir" "$target_dir/"
        echo "  ✓ Wildcards copied successfully to $target_dir/mbe"
    else
        echo "  ⚠ Source directory not found: $source_dir"
        echo "  → Trying alternative path..."
        # Try alternative path if comfy-tagcomplete is in workspace
        local alt_source_dir="/workspace/comfy-tagcomplete/wildcards/mbe"
        if [[ -d "$alt_source_dir" ]]; then
            echo "  → Found at alternative path, copying..."
            cp -r "$alt_source_dir" "$target_dir/"
            echo "  ✓ Wildcards copied successfully from alternative path"
        else
            echo "  ⚠ Source directory not found at: $alt_source_dir"
        fi
    fi
}

function provisioning_get_files() {
    if [[ -z $2 ]]; then return 1; fi
    
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    echo "Downloading ${#arr[@]} file(s) to $dir..."
    local count=0
    for url in "${arr[@]}"; do
        ((count++))
        echo "[$count/${#arr[@]}] Downloading: $(basename "$url")"
        provisioning_download "${url}" "${dir}"
        echo "  ✓ Download completed"
    done
    echo "All files downloaded successfully!"
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Application will start now\n\n"
}

# Allow user to disable provisioning if they started with a script they didn't want
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi