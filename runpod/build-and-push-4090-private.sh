#!/bin/bash

# ComfyUI-QwenVL-Mod v2.2.0 RTX 4090 Build Script
# Automates building and pushing Docker image with CUDA 12.4 for RTX 4090

set -e

# Configuration
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod-4090"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🚀 Building ComfyUI-QwenVL-Mod v2.2.0 RTX 4090 Docker Image"
echo "======================================================"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Features: WAN 2.2 workflows, GGUF support, CUDA 12.4 optimized"
echo "GPU: RTX 4090 compatible"
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
echo "This may take 10-15 minutes..."
docker buildx build --builder desktop-linux --platform linux/amd64 -f Dockerfile.4090 -t "${FULL_IMAGE_NAME}" --load --no-cache .

# Push to Docker Hub
echo "📤 Pushing image to Docker Hub..."
docker push "${FULL_IMAGE_NAME}"

echo ""
echo "✅ Build and push completed successfully!"
echo ""
echo "📋 Image Details:"
echo "  Latest tag:  ${FULL_IMAGE_NAME}"
echo "  Dockerfile:   Dockerfile.4090"
echo "  CUDA Version: 12.4"
echo "  GPU Target:   RTX 4090"
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
echo "💡 Port mappings for RTX 4090:"
echo "  8080 → FileBrowser"
echo "  8888 → JupyterLab (Terminal included)"
echo "  8188 → ComfyUI"
echo ""
echo "🔗 Useful links:"
echo "  Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
echo "  Repository: https://github.com/huchukato/ComfyUI-QwenVL-Mod"
echo ""
echo "🎯 Differences from RTX 5090 version:"
echo "  • CUDA 12.4 (vs 12.8)"
echo "  • llama-cpp-python from source (vs pre-built wheel)"
echo "  • Optimized for RTX 4090 compatibility"
echo "  • Same features and workflows"
