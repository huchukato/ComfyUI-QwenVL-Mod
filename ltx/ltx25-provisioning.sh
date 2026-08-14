#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI
APT_INSTALL="${APT_INSTALL:-apt-get install -y}"

APT_PACKAGES=(
    "aria2"
)

PIP_PACKAGES=(
    "--upgrade --force-reinstall --no-cache-dir https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.45-cu131-linux-20260801/llama_cpp_python-0.3.45+cu131-cp312-cp312-linux_x86_64.whl"
    "huggingface_hub"
    "hf-transfer"
    "sageattention"
    "kornia==0.8.2"
    "tensorrt-cu13==10.15.1.29"
    "tensorrt-cu13-bindings==10.15.1.29"
    "tensorrt-cu13-libs==10.15.1.29"
)

NODES=(
    "https://github.com/huchukato/comfy-tagcomplete"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod"
    "https://github.com/BobRandomNumber/ComfyUI-Crystools-MonitorOnly"
    "https://github.com/Lightricks/ComfyUI-LTXVideo"
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
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/ltx/LTX25-I2VA-Qwen3VL.json"
    "https://github.com/huchukato/ComfyUI-QwenVL-Mod/raw/main/ltx/LTX25-FL2VA-Qwen3VL.json"
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
)

TEXT_ENCODERS=(
)

CONTROLNET_MODELS=(
)

# LTX 2.5 models: subdir|name|url|min_size_bytes (native uncensored, from huchukato/pimp-my-wan mirror)
LTX_MODELS=(
    # ── LTX 2.5 (native uncensored) ──
    "diffusion_models|ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|21000000000"
    "text_encoders|gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|15000000000"
    "text_encoders|gemma4_e2b_it_bf16.safetensors|https://huggingface.co/TrevorJS/gemma-4-E2B-it-uncensored/resolve/main/model.safetensors|10000000000"
    "vae|ltx-2.5-video-vae-bf16.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/vae/ltx-2.5-video-vae-bf16.safetensors|1400000000"
    "vae|ltx-2.5-audio-vae-bf16.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/vae/ltx-2.5-audio-vae-bf16.safetensors|350000000"
    "latent_upscale_models|ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|990000000"
    "latent_upscale_models|ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors|250000000"
    "model_patches|ltx-2.5-duration-head-bf16.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/model_patches/ltx-2.5-duration-head-bf16.safetensors|3800000"
)

function provisioning_force_comfyui_version() {
    local repo_dir="$1"
    local label="$2"
    local tag="v0.31.0"

    if [ ! -d "$repo_dir/.git" ]; then
        echo "⚠️  $label has no .git directory, skipping version force"
        return 0
    fi

    echo "🔧 Ensuring $label is on v0.31.0 (LTX 2.5 requirement)..."
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

function download_ltx_model() {
    local base_dir="$1" subdir="$2" name="$3" url="$4" min_size="$5"
    local dest="$base_dir/$subdir/$name"
    local max_retries=5

    # Parse Hugging Face repo_id and relative repo path from the URL.
    local repo_id repo_path
    repo_id=$(echo "$url" | awk -F/ '{print $4"/"$5}')
    repo_path=$(echo "$url" | sed -E 's#https?://[^/]+/[^/]+/[^/]+/resolve/main/(.+)#\1#')

    # Use 'hf' command (newer huggingface_hub) or fall back to 'huggingface-cli'
    local hf_cmd="hf"
    command -v hf >/dev/null 2>&1 || hf_cmd="huggingface-cli"

    for attempt in $(seq 1 $max_retries); do
        if [ -f "$dest" ] || [ -L "$dest" ]; then
            local size
            size=$(stat -L -c%s "$dest" 2>/dev/null || stat -L -f%z "$dest" 2>/dev/null || echo 0)
            if [ "$size" -ge "$min_size" ]; then
                echo "✅ $name already present ($size bytes >= $min_size), skipping"
                return 0
            else
                echo "⚠️  $name incomplete ($size bytes < $min_size), retrying (attempt $attempt/$max_retries)"
            fi
        else
            echo "📥 Downloading $name (attempt $attempt/$max_retries)..."
        fi

        # Download to a temp dir to avoid path nesting issues.
        local tmp_dir="$base_dir/.tmp_download_${name//\//_}"
        rm -rf "$tmp_dir"
        mkdir -p "$tmp_dir" "$(dirname "$dest")"
        export HF_HUB_ENABLE_HF_TRANSFER=1
        export HF_XET_HIGH_PERFORMANCE=1
        # hf: resume is automatic; huggingface-cli: needs --resume-download
        local resume_flag=""
        [ "$hf_cmd" = "huggingface-cli" ] && resume_flag="--resume-download"
        if $hf_cmd download "$repo_id" "$repo_path" \
                --local-dir "$tmp_dir" \
                $resume_flag 2>&1; then
            local downloaded_path="$tmp_dir/$repo_path"
            if [ -f "$downloaded_path" ] || [ -L "$downloaded_path" ]; then
                if [ -L "$downloaded_path" ]; then
                    ln -sf "$(readlink -f "$downloaded_path")" "$dest"
                else
                    mv -f "$downloaded_path" "$dest"
                fi
                rm -rf "$tmp_dir"
                local size
                size=$(stat -L -c%s "$dest" 2>/dev/null || stat -L -f%z "$dest" 2>/dev/null || echo 0)
                if [ "$size" -ge "$min_size" ]; then
                    echo "✅ $name downloaded successfully ($size bytes)"
                    return 0
                else
                    echo "⚠️  $name downloaded but size $size < $min_size, will retry"
                fi
            else
                echo "⚠️  $name not found at $downloaded_path after download, will retry"
                rm -rf "$tmp_dir"
            fi
        else
            echo "⚠️  $hf_cmd failed for $name (attempt $attempt), retrying in $((attempt*10))s..."
            rm -rf "$tmp_dir"
        fi

        [ "$attempt" -lt "$max_retries" ] && sleep $((attempt * 10))
    done

    echo "❌ FAILED: $name could not be downloaded after $max_retries attempts"
    return 1
}

function provisioning_get_ltx_models() {
    local base_dir="${WORKSPACE:-/workspace}/ComfyUI/models"
    local ready_marker="${WORKSPACE:-/workspace}/ComfyUI/main.py"

    local all_complete=true
    for entry in "${LTX_MODELS[@]}"; do
        IFS='|' read -r subdir name url min_size <<< "$entry"
        local dest="$base_dir/$subdir/$name"
        local size
        size=$(stat -L -c%s "$dest" 2>/dev/null || stat -L -f%z "$dest" 2>/dev/null || echo 0)
        if [ "$size" -lt "$min_size" ]; then
            all_complete=false
            break
        fi
    done
    if [ "$all_complete" = true ]; then
        echo "✅ All LTX 2.5 models already complete, no download needed"
        return 0
    fi

    mkdir -p "$base_dir"/{checkpoints,text_encoders,loras/ltx23,latent_upscale_models,vae,diffusion_models,model_patches}

    echo "📥 === LTX 2.5 model download started (PID $$) ==="
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
    for entry in "${LTX_MODELS[@]}"; do
        IFS='|' read -r subdir name url min_size <<< "$entry"
        download_ltx_model "$base_dir" "$subdir" "$name" "$url" "$min_size" || failures=$((failures + 1))
    done

    echo "📦 === LTX 2.5 model download finished ($failures failures) ==="
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
        echo "✅ Created $args_file with LTX 2.5 / QwenVL-Mod optimized args"
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
    
    echo "🔧 Ensuring ComfyUI is on v0.31.0 (LTX 2.5 requirement)..."
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
    
    echo "🧬 Starting LTX 2.5 model download in background..."
    provisioning_get_ltx_models
    
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
