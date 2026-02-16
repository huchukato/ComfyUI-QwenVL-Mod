#!/bin/bash

# ComfyUI-QwenVL-Mod v2.2.0 RTX 5090 Local Build Script
# Builds Docker image locally for local use only

set -e

# Configuration
IMAGE_NAME="comfyui-qwenvl-local"
TAG="5090"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🏗️  Building ComfyUI-QwenVL-Mod v2.2.0 RTX 5090 Docker Image Locally"
echo "=================================================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo "Features: WAN 2.2 workflows, GGUF support, CUDA 12.8 optimized"
echo "GPU: RTX 5090 compatible"
echo "Use: Local Docker deployment only"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build the image
echo "🏗️  Building Docker image for x86_64 (CUDA 12.8)..."
echo "This may take 10-15 minutes..."
docker buildx build --builder desktop-linux --platform linux/amd64 -f Dockerfile -t "${FULL_IMAGE_NAME}" --load --no-cache .

echo ""
echo "✅ Local build completed successfully!"
echo ""
echo "📋 Image Details:"
echo "  Local tag:  ${FULL_IMAGE_NAME}"
echo "  Dockerfile:  Dockerfile"
echo "  CUDA Version: 12.8"
echo "  GPU Target:   RTX 5090"
echo ""
echo "🚀 Ready for local Docker deployment!"
echo ""
echo "📝 To run locally:"
echo "  docker run -d --gpus all \\"
echo "    -p 8080:8080 -p 8888:8888 -p 8188:8188 \\"
echo "    -v \$(pwd)/output:/ComfyUI/output \\"
echo "    ${FULL_IMAGE_NAME}"
echo ""
echo "💡 Port mappings:"
echo "  8080 → FileBrowser"
echo "  8888 → JupyterLab (Terminal included)"
echo "  8188 → ComfyUI"
echo ""
echo "🔗 For RunPod deployment, use the pre-built image:"
echo "  docker pull huchukato/comfyui-qwenvl-runpod"
echo "  Then deploy on RunPod with RTX 5090"
