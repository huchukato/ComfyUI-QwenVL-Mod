#!/bin/bash

# ComfyUI-QwenVL-Mod v2.0.9 SLIM Build Script
# Optimized for smaller image size (~15-20GB)

set -e

# Configuration
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="slim"
VERSION_TAG="v2.0.9-slim"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
VERSION_IMAGE_NAME="${IMAGE_NAME}:${VERSION_TAG}"

echo "🚀 Building ComfyUI-QwenVL-Mod v2.0.9 SLIM Docker Image"
echo "======================================================"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Features: Bypass Mode, WAN 2.2 workflows, GGUF support"
echo "Size target: ~15-20GB (vs 45GB full)"
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

# Build the slim image
echo "🏗️  Building SLIM Docker image for x86_64..."
echo "This may take 8-12 minutes (smaller than full build)..."
docker buildx build --builder desktop-linux --platform linux/amd64 -f Dockerfile.slim -t "${FULL_IMAGE_NAME}" -t "${VERSION_IMAGE_NAME}" --load .

# Push to Docker Hub
echo "📤 Pushing SLIM images to Docker Hub..."
docker push "${FULL_IMAGE_NAME}"
docker push "${VERSION_IMAGE_NAME}"

# Create additional tags
echo "🏷️  Creating additional tags..."
CURRENT_DATE=$(date +%Y%m%d)
DATE_TAG="${IMAGE_NAME}:slim-${CURRENT_DATE}"
docker tag "${FULL_IMAGE_NAME}" "${DATE_TAG}"
docker push "${DATE_TAG}"

echo ""
echo "✅ SLIM build and push completed successfully!"
echo ""
echo "📋 SLIM Image Details:"
echo "  Slim tag:    ${FULL_IMAGE_NAME}"
echo "  Version tag: ${VERSION_IMAGE_NAME}"
echo "  Date tag:    ${DATE_TAG}"
echo ""
echo "📊 Size Comparison:"
echo "  Full image:  ~45GB"
echo "  Slim image:  ~15-20GB"
echo "  Savings:     ~25-30GB"
echo ""
echo "🚀 Ready for RunPod deployment (smaller footprint)!"
echo ""
echo "📝 Next steps:"
echo "  1. Go to RunPod Console"
echo "  2. Create new endpoint"
echo "  3. Use image: ${FULL_IMAGE_NAME}"
echo "  4. Select GPU: RTX 5090"
echo "  5. Set container port: 8188"
echo "  6. Deploy! 🎉"
echo ""
echo "💡 Note: SLIM version includes essential nodes only"
echo "🔗 Useful links:"
echo "  Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
echo "  Repository: https://github.com/huchukato/ComfyUI-QwenVL-Mod"
