#!/bin/bash

# ComfyUI-QwenVL-Mod v2.1.0 Build Script
# Automates building and pushing Docker image with Keep Last Prompt to Docker Hub

set -e

# Configuration
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod-5090"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🚀 Building ComfyUI-QwenVL-Mod v2.2.0 RTX 5090 Docker Image"
echo "======================================================"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Features: WAN 2.2 workflows, GGUF support, CUDA 12.8"
echo "GPU: RTX 5090 optimized"
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
echo "🏗️  Building Docker image for x86_64..."
echo "This may take 10-15 minutes..."
docker buildx build --builder desktop-linux --platform linux/amd64 -t "${FULL_IMAGE_NAME}" --load --no-cache .

# Push to Docker Hub
echo "📤 Pushing image to Docker Hub..."
docker push "${FULL_IMAGE_NAME}"

echo ""
echo "✅ Build and push completed successfully!"
echo ""
echo "📋 Image Details:"
echo "  Latest tag:  ${FULL_IMAGE_NAME}"
echo ""
echo "🚀 Ready for RunPod deployment!"
echo ""
echo "📝 Next steps:"
echo "  1. Go to RunPod Console"
echo "  2. Create new endpoint"
echo "  3. Use image: ${FULL_IMAGE_NAME}"
echo "  4. Select GPU: RTX 5090"
echo "  5. Set container port: 8188"
echo "  6. Deploy! 🎉"
echo ""
echo "💡 Note: Local testing disabled for faster builds"
echo "🔗 Useful links:"
echo "  Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
echo "  Repository: https://github.com/huchukato/ComfyUI-QwenVL-Mod"
