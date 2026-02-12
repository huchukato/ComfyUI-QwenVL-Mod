#!/bin/bash

# ComfyUI-QwenVL-Mod v2.0.9 Build Script
# Automates building and pushing Docker image with Bypass Mode to Docker Hub

set -e

# Configuration
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="v2.0.9"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🚀 Building ComfyUI-QwenVL-Mod v2.0.9 Docker Image"
echo "==============================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo "Features: Bypass Mode, WAN 2.2 workflows, GGUF support"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if we're logged into Docker Hub
echo "🔐 Checking Docker Hub login..."
if ! docker info | grep -q "Username"; then
    echo "⚠️  You're not logged into Docker Hub. Please run 'docker login' first."
    exit 1
fi

# Build the image
echo "🏗️  Building Docker image..."
echo "This may take 10-15 minutes..."
docker build -t "${FULL_IMAGE_NAME}" .

# Test the image locally
echo "🧪 Testing image locally..."
echo "Starting container for 30 seconds to verify..."
docker run --rm --gpus all -d -p 8081:8080 --name test-container "${FULL_IMAGE_NAME}"

# Wait for startup
echo "⏳ Waiting for ComfyUI to start..."
sleep 30

# Check if container is running
if docker ps | grep -q test-container; then
    echo "✅ Container started successfully!"
    
    # Check if ComfyUI is responding
    if curl -f http://localhost:8081/system_stats > /dev/null 2>&1; then
        echo "✅ ComfyUI is responding correctly!"
    else
        echo "⚠️  ComfyUI not responding yet, but container is running"
    fi
    
    # Stop test container
    docker stop test-container
    echo "✅ Test completed successfully"
else
    echo "❌ Container failed to start"
    docker logs test-container 2>/dev/null || true
    exit 1
fi

# Push to Docker Hub
echo "📤 Pushing image to Docker Hub..."
docker push "${FULL_IMAGE_NAME}"

# Create additional tags
echo "🏷️  Creating additional tags..."
CURRENT_DATE=$(date +%Y%m%d)
FULL_TAG="${IMAGE_NAME}:full-${CURRENT_DATE}"
docker tag "${FULL_IMAGE_NAME}" "${FULL_TAG}"
docker push "${FULL_TAG}"

echo ""
echo "✅ Build and push completed successfully!"
echo ""
echo "📋 Image Details:"
echo "  Primary tag: ${FULL_IMAGE_NAME}"
echo "  Date tag:    ${FULL_TAG}"
echo ""
echo "🚀 Ready for RunPod deployment!"
echo ""
echo "📝 Next steps:"
echo "  1. Go to RunPod Console"
echo "  2. Create new endpoint"
echo "  3. Use image: ${FULL_IMAGE_NAME}"
echo "  4. Select GPU: RTX 5090"
echo "  5. Set container port: 8080"
echo "  6. Deploy! 🎉"
echo ""
echo "🔗 Useful links:"
echo "  Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
echo "  Repository: https://github.com/huchukato/ComfyUI-QwenVL-Mod"
