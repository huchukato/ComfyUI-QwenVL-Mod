#!/bin/bash

# ComfyUI-QwenVL-Mod RunPod Light Build Script
# Builds lightweight Docker image without pre-downloaded models

set -e

# Configuration
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
LIGHT_TAG="${IMAGE_NAME}:light"
FULL_TAG="${IMAGE_NAME}:latest"

echo "🚀 Building ComfyUI-QwenVL-Mod RunPod Docker Image (Lightweight)"
echo "==============================================================="
echo "Light Image: ${LIGHT_TAG}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build the light image
echo "🏗️  Building lightweight Docker image..."
echo "This will take ~5-8 minutes (much faster than full version)..."
docker build -f Dockerfile.light -t "${LIGHT_TAG}" .

# Test the image locally
echo "🧪 Testing image locally..."
echo "Starting container for 30 seconds to verify..."
docker run --rm --gpus all -d -p 8081:8080 --name test-light-container "${LIGHT_TAG}"

# Wait for startup
echo "⏳ Waiting for ComfyUI to start..."
sleep 30

# Check if container is running
if docker ps | grep -q test-light-container; then
    echo "✅ Container started successfully!"
    
    # Check if ComfyUI is responding
    if curl -f http://localhost:8081/system_stats > /dev/null 2>&1; then
        echo "✅ ComfyUI is responding correctly!"
    else
        echo "⚠️  ComfyUI not responding yet, but container is running"
    fi
    
    # Stop test container
    docker stop test-light-container
    echo "✅ Test completed successfully"
else
    echo "❌ Container failed to start"
    docker logs test-light-container 2>/dev/null || true
    exit 1
fi

# Push to Docker Hub
echo "📤 Pushing light image to Docker Hub..."
docker push "${LIGHT_TAG}"

echo ""
echo "✅ Light build and push completed successfully!"
echo ""
echo "📋 Image Details:"
echo "  Light tag:   ${LIGHT_TAG}"
echo "  Full tag:    ${FULL_TAG} (if exists)"
echo ""
echo "💡 Light version benefits:"
echo "  - Faster build time (~5-8 min vs 15+ min)"
echo "  - Smaller image size (~5-8GB vs 22GB)"
echo "  - Models downloaded on-demand"
echo "  - Faster container startup"
echo ""
echo "🚀 Ready for RunPod deployment!"
echo ""
echo "📝 Next steps:"
echo "  1. Go to RunPod Console"
echo "  2. Create new endpoint"
echo "  3. Use image: ${LIGHT_TAG}"
echo "  4. Select GPU: RTX 5090"
echo "  5. Set container port: 8080"
echo "  6. Deploy! 🎉"
echo ""
echo "🔗 Useful links:"
echo "  Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
echo "  Repository: https://github.com/huchukato/ComfyUI-QwenVL-Mod"
