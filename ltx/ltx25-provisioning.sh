#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI

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

# LTX 2.5 large models downloaded via hf/huggingface-cli (format: subdir|name|url|min_size_bytes)
LTX_MODELS=(
    "diffusion_models|ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|21000000000"
    "text_encoders|gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|15000000000"
    "text_encoders|gemma4_e2b_it_bf16.safetensors|https://huggingface.co/TrevorJS/gemma-4-E2B-it-uncensored/resolve/main/model.safetensors|10000000000"
    "vae|ltx-2.5-video-vae-bf16.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/vae/ltx-2.5-video-vae-bf16.safetensors|1400000000"
    "vae|ltx-2.5-audio-vae-bf16.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/vae/ltx-2.5-audio-vae-bf16.safetensors|350000000"
    "latent_upscale_models|ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|990000000"
    "latent_upscale_models|ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors|250000000"
    "model_patches|ltx-2.5-duration-head-bf16.safetensors|https://huggingface.co/huchukato/pimp-my-wan/resolve/main/LTX/model_patches/ltx-2.5-duration-head-bf16.safetensors|3800000"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    provisioning_print_header
    echo "🚀 Starting provisioning process..."

    echo "📦 Installing APT packages..."
    provisioning_get_apt_packages

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

    echo "🎬 Downloading LTX 2.5 models (large files via hf)..."
    download_ltx_models

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
                local branch
                branch=$(git -C "$path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
                if git -C "$path" pull --ff-only origin "$branch" 2>/dev/null; then
                    echo "  ✅ $dir updated"
                else
                    echo "  ⚠️  $dir pull failed, resetting to origin/$branch..."
                    git -C "$path" fetch origin "$branch" 2>/dev/null && \
                        git -C "$path" reset --hard "origin/$branch" 2>/dev/null || \
                        echo "  ⚠️  $dir reset failed, leaving as-is"
                fi
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

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_download() {
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" --content-disposition -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget --content-disposition -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

function download_ltx_models() {
    local base_dir="${COMFYUI_DIR}/models"
    mkdir -p "$base_dir"/{diffusion_models,text_encoders,vae,latent_upscale_models,model_patches}

    local hf_cmd="hf"
    command -v hf >/dev/null 2>&1 || hf_cmd="huggingface-cli"

    for entry in "${LTX_MODELS[@]}"; do
        IFS='|' read -r subdir name url min_size <<< "$entry"
        local dest="$base_dir/$subdir/$name"

        if [ -f "$dest" ] || [ -L "$dest" ]; then
            local size
            size=$(stat -L -c%s "$dest" 2>/dev/null || stat -L -f%z "$dest" 2>/dev/null || echo 0)
            if [ "$size" -ge "$min_size" ]; then
                echo "✅ $name already present ($size bytes >= $min_size), skipping"
                continue
            fi
        fi

        echo "📥 Downloading $name ..."
        local repo_id repo_path tmp_dir
        repo_id=$(echo "$url" | awk -F/ '{print $4"/"$5}')
        repo_path=$(echo "$url" | sed -E 's#https?://[^/]+/[^/]+/[^/]+/resolve/main/(.+)#\1#')
        tmp_dir="$base_dir/.tmp_download_${name//\//_}"
        rm -rf "$tmp_dir"
        mkdir -p "$tmp_dir"

        export HF_HUB_ENABLE_HF_TRANSFER=1
        export HF_XET_HIGH_PERFORMANCE=1
        local resume_flag=""
        [ "$hf_cmd" = "huggingface-cli" ] && resume_flag="--resume-download"

        if $hf_cmd download "$repo_id" "$repo_path" --local-dir "$tmp_dir" $resume_flag; then
            local downloaded_path="$tmp_dir/$repo_path"
            if [ -f "$downloaded_path" ] || [ -L "$downloaded_path" ]; then
                mkdir -p "$(dirname "$dest")"
                mv -f "$downloaded_path" "$dest"
                rm -rf "$tmp_dir"
                local size
                size=$(stat -L -c%s "$dest" 2>/dev/null || stat -L -f%z "$dest" 2>/dev/null || echo 0)
                echo "✅ $name downloaded successfully ($size bytes)"
            else
                echo "⚠️  $name not found after download"
                rm -rf "$tmp_dir"
            fi
        else
            echo "❌ $hf_cmd failed for $name"
            rm -rf "$tmp_dir"
        fi
    done
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
