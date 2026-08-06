#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI
APT_INSTALL="${APT_INSTALL:-apt-get install -y}"

APT_PACKAGES=(
    "aria2"
)

PIP_PACKAGES=(
    "--upgrade --force-reinstall --no-cache-dir https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.45-cu131-linux-20260801/llama_cpp_python-0.3.45+cu131-cp312-cp312-linux_x86_64.whl"
    "sageattention"
    "tensorrt-cu13==10.15.1.29"
    "tensorrt-cu13-bindings==10.15.1.29"
    "tensorrt-cu13-libs==10.15.1.29"
)

NODES=(
    "https://github.com/huchukato/comfy-tagcomplete"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
    "https://github.com/huchukato/ComfyUI-Upscaler-TensorRT-Auto"
    "https://github.com/huchukato/ComfyUI-HuggingFace"
    "https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler"
    "https://github.com/ltdrdata/was-node-suite-comfyui"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/ashtar1984/comfyui-find-perfect-resolution"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/MoonGoblinDev/Civicomfy"
)

WORKFLOWS=(
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/minimax/MiniMaxH3-I2VA-Qwen3VL.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/minimax/MiniMaxH3-FL2VA-Qwen3VL.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/minimax/MiniMaxH3-T2VA-Qwen3VL.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/minimax/MiniMaxH3-R2VA-Qwen3VL.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/vastai/workflows/PMP-LoRaStack-Upscale-Wildcards.json"
)

CHECKPOINT_MODELS=(
)

UNET_MODELS=(
   
)

LORA_MODELS=(
            
)

VAE_MODELS=(
)

ESRGAN_MODELS=(
    "https://huggingface.co/huchukato/favs/resolve/main/ESRGAN/2xLexicaRRDBNet.pth"
    "https://huggingface.co/huchukato/favs/resolve/main/ESRGAN/2xLexicaRRDBNet_Sharp.pth"
)

TEXT_ENCODERS=(
)

MINIMAX_MODELS=(
    "vae|minimax_h3_video_vae_fp16.safetensors|https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors|5200000000"
    "vae|minimax_h3_audio_vae_fp32.safetensors|https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors|600000000"
    "diffusion_models|10Eros_Max_H3_FL2VA-INT8-ConvRot-HQ.safetensors|https://huggingface.co/DmitryDB/MiniMax-H3-10Eros-Max-Quants/resolve/main/FL2VA/10Eros_Max_H3_FL2VA-INT8-ConvRot-HQ.safetensors|23523980600"
    "text_encoders|qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors|https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot/resolve/main/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors|26363476151"
)

CONTROLNET_MODELS=(
)

function provisioning_force_comfyui_version() {
    local repo_dir="$1"
    local label="$2"
    local tag="v0.30.0"

    if [ ! -d "$repo_dir/.git" ]; then
        echo "⚠️  $label has no .git directory, skipping version force"
        return 0
    fi

    echo "🔧 Ensuring $label is on $tag (MiniMax H3 requirement)..."
    if timeout 60 git -C "$repo_dir" fetch --tags --force origin 2>/dev/null; then
        local current_hash target_hash
        current_hash=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo "unknown")
        if timeout 30 git -C "$repo_dir" -c advice.detachedHead=false checkout -f "$tag" 2>/dev/null; then
            target_hash=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo "unknown")
            echo "✅ $label forced to $tag ($current_hash -> $target_hash)"
            if [ -f "$repo_dir/requirements.txt" ]; then
                echo "🔄 Re-installing ComfyUI requirements..."
                pip install --root-user-action=ignore --no-cache-dir -r "$repo_dir/requirements.txt" 2>&1 | tail -n 20
            fi
        else
            echo "❌ $label checkout $tag failed"
        fi
    else
        echo "⚠️  $label fetch tags failed/timed out (offline?), leaving current version"
    fi
}

function download_minimax_model() {
    local base_dir="$1" subdir="$2" name="$3" url="$4" min_size="$5"
    local dest="$base_dir/$subdir/$name"
    local max_retries=5
    local hf_auth=()

    if [ -n "$HF_TOKEN" ] && [[ "$url" == *"huggingface.co"* ]]; then
        hf_auth=(--header="Authorization: Bearer $HF_TOKEN")
    fi

    for attempt in $(seq 1 $max_retries); do
        if [ -f "$dest" ]; then
            local size
            size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
            if [ "$size" -ge "$min_size" ]; then
                echo "✅ $name already present ($size bytes >= $min_size), skipping"
                return 0
            else
                echo "⚠️  $name incomplete ($size bytes < $min_size), resuming (attempt $attempt/$max_retries)"
            fi
        else
            echo "📥 Downloading $name (attempt $attempt/$max_retries)..."
        fi

        local downloader_pid
        if command -v aria2c >/dev/null 2>&1; then
            # aria2c: 8 connessioni parallele, split in 8 segmenti, resume
            aria2c -q -c -x 8 -s 8 --max-connection-per-server=8 --min-split-size=10M \
                --auto-file-renaming=false --allow-overwrite=true --continue=true \
                --dir="$base_dir/$subdir" --out="$name" "${hf_auth[@]}" "$url" &
            downloader_pid=$!
        else
            wget -q -c --tries=3 --timeout=120 "${hf_auth[@]}" "$url" -O "$dest" &
            downloader_pid=$!
        fi

        (
            while kill -0 $downloader_pid 2>/dev/null; do
                if [ -f "$dest" ]; then
                    local s
                    s=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
                    local pct=$(( s * 100 / min_size ))
                    local gb=$(( s / 1073741824 ))
                    local total_gb=$(( min_size / 1073741824 ))
                    echo "📊 PROGRESS: $name — ${gb}GB / ${total_gb}GB (${pct}%)"
                fi
                sleep 10
            done
        ) &
        local monitor_pid=$!

        if wait $downloader_pid; then
            kill $monitor_pid 2>/dev/null; wait $monitor_pid 2>/dev/null
            local size
            size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
            if [ "$size" -ge "$min_size" ]; then
                echo "✅ $name downloaded successfully ($size bytes)"
                return 0
            else
                echo "⚠️  $name downloaded but size $size < $min_size, will retry"
            fi
        else
            kill $monitor_pid 2>/dev/null; wait $monitor_pid 2>/dev/null
            echo "⚠️  Download failed for $name (attempt $attempt), retrying in $((attempt*10))s..."
        fi

        [ "$attempt" -lt "$max_retries" ] && sleep $((attempt * 10))
    done

    echo "❌ FAILED: $name could not be downloaded after $max_retries attempts"
    return 1
}

function provisioning_get_minimax_models() {
    local base_dir="${WORKSPACE:-/workspace}/ComfyUI/models"
    local ready_marker="${WORKSPACE:-/workspace}/ComfyUI/main.py"

    local all_complete=true
    for entry in "${MINIMAX_MODELS[@]}"; do
        IFS='|' read -r subdir name url min_size <<< "$entry"
        local dest="$base_dir/$subdir/$name"
        local size
        size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
        if [ "$size" -lt "$min_size" ]; then
            all_complete=false
            break
        fi
    done
    if [ "$all_complete" = true ]; then
        echo "✅ All MiniMax H3 models already complete, no download needed"
        return 0
    fi

    mkdir -p "$base_dir"/{vae,diffusion_models,text_encoders}

    echo "� === MiniMax H3 model download started (PID $$) ==="
    echo "⏳ Waiting for ComfyUI ready marker..."
    for i in $(seq 1 120); do
        [ -f "$ready_marker" ] && break
        sleep 5
    done

    if [ ! -f "$ready_marker" ]; then
        echo "❌ ERROR: ComfyUI ready marker not found after 600s, aborting"
        return 1
    fi

    echo "✅ ComfyUI ready, base dir: $base_dir"

    local failures=0
    for entry in "${MINIMAX_MODELS[@]}"; do
        IFS='|' read -r subdir name url min_size <<< "$entry"
        download_minimax_model "$base_dir" "$subdir" "$name" "$url" "$min_size" || failures=$((failures + 1))
    done

    echo "📦 === MiniMax H3 model download finished ($failures failures) ==="
}

function provisioning_configure_args() {
    local args_file="${WORKSPACE:-/workspace}/ComfyUI/comfyui_args.txt"
    if [ ! -f "$args_file" ]; then
        mkdir -p "$(dirname "$args_file")"
        cat > "$args_file" <<'EOF'
--disable-auto-launch
--fast fp16_accumulation
--use-sage-attention
--reserve-vram 2
--cuda-malloc
--async-offload
EOF
        echo "✅ Created $args_file with MiniMax H3 / QwenVL-Mod optimized args"
    else
        echo "ℹ️  $args_file already exists, leaving untouched"
    fi
}

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    provisioning_print_header
    echo "🚀 Starting provisioning process..."
    
    echo "📦 Installing APT packages..."
    provisioning_get_apt_packages
    
    echo "🔧 Ensuring ComfyUI is on v0.30.0 (MiniMax H3 requirement)..."
    provisioning_force_comfyui_version "${COMFYUI_DIR}" "ComfyUI"
    
    echo "🔧 Installing custom nodes..."
    provisioning_get_nodes
    
    echo "📦 Installing PIP packages..."
    provisioning_get_pip_packages
    
    echo "📁 Downloading workflows..."
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
    
    echo "🧬 Starting MiniMax H3 model download in background..."
    provisioning_get_minimax_models
    
    echo "🎛️ Configuring ComfyUI arguments..."
    provisioning_configure_args
    
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
        
    done
    echo "All nodes processed successfully!"
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

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\\.)?huggingface\\.co(/|$|\\?) ]]; then
        auth_token="$HF_TOKEN"
    elif 
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\\.)?civitai\\.com(/|$|\\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" --content-disposition -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget --content-disposition -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

# Allow user to disable provisioning if they started with a script they didn't want
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
