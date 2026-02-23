#!/bin/bash

# ComfyUI-QwenVL-Mod v2.2.0 RunPod Official CUDA 12.4 Build Script
# Automates building and pushing Docker image with official RunPod approach (no conda, CUDA 12.4)

set -e

# Configuration
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="official-12.4"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🚀 Building ComfyUI-QwenVL-Mod v2.2.0 RunPod Official CUDA 12.4 Docker Image"
echo "==========================================================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo "Features: WAN 2.2 workflows, GGUF support, CUDA 12.4 optimized"
echo "Approach: Official RunPod (multi-stage, no conda, Python 3.12)"
echo "GPU: RTX 4090 compatible (CUDA 12.4)"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if we're logged into Docker Hub (optional check)
echo "🔐 Checking Docker Hub login..."
if ! docker login --get-identity > /dev/null 2>&1; then
    echo "⚠️  Docker Hub login check failed, but continuing..."
    echo "💡 If push fails later, run: docker login"
else
    echo "✅ Docker Hub login verified"
fi

# Build the image
echo "🏗️  Building Docker image for x86_64 (CUDA 12.4)..."
echo "Using official RunPod multi-stage approach..."
echo "This may take 10-15 minutes..."
docker buildx build --builder desktop-linux --platform linux/amd64 -f Dockerfile.runpod-official-12.4 -t "${FULL_IMAGE_NAME}" --load --no-cache .

# Push to Docker Hub
echo "📤 Pushing image to Docker Hub..."
docker push "${FULL_IMAGE_NAME}"

echo ""
echo "✅ Build and push completed successfully!"
echo ""
echo "📋 Image Details:"
echo "  Tag:          ${FULL_IMAGE_NAME}"
echo "  Dockerfile:   Dockerfile.runpod-official-12.4"
echo "  CUDA Version: 12.4"
echo "  Python:       3.12 (system, no conda)"
echo "  Approach:     Official RunPod multi-stage"
echo "  GPU Target:   RTX 4090 compatible"
echo ""
echo "🚀 Ready for RunPod deployment!"
echo ""
echo "📝 Next steps:"
echo "  1. Go to RunPod Console"
echo "  2. Create new endpoint"
echo "  3. Use image: ${FULL_IMAGE_NAME}"
echo "  4. Select GPU: RTX 4090"
echo "  5. Set container port: 8188"
echo "  6. Open additional ports: 8080, 8888"
echo "  7. Deploy! 🎉"
echo ""
echo "💡 Port mappings:"
echo "  8080 → FileBrowser"
echo "  8888 → JupyterLab (Terminal included)"
echo "  8188 → ComfyUI"
echo ""
echo "🔧 Environment Variables (optional):"
echo "  COMFYUI_ARGS: Override default ComfyUI arguments"
echo "  Default: --listen 0.0.0.0 --port 8188 --disable-auto-launch --enable-cors-header --fast fp16_accumulation --use-sage-attention --reserve-vram 2 --cuda-malloc --async-offload"
echo ""
echo "📊 Benefits vs conda version:"
echo "  ✅ Smaller image size"
echo "  ✅ Faster startup"
echo "  ✅ Official RunPod compatibility"
echo "  ✅ CUDA 12.4 optimized for RTX 4090"
