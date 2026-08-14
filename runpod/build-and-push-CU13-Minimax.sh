#!/bin/bash

# Build and Push Script for ComfyUI-QwenVL-Mod (RunPod, CUDA 13.0, MiniMax H3 edition)
# Based on runpod/comfyui:cuda13.0 template

set -e

echo "🐳 Building ComfyUI-QwenVL-Mod Docker image (RunPod, CUDA 13.0, MiniMax H3)..."

# Build variables
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="cu13-minimax"
DOCKERFILE="Dockerfile.CU13-Minimax"
PLATFORM="linux/amd64"

# ComfyUI core version to bake into the image
COMFYUI_VERSION="${COMFYUI_VERSION:-v0.32.0}"

# Check for latest upstream ComfyUI release tag
LATEST_COMFYUI_VERSION=$(curl -s "https://api.github.com/repos/comfyanonymous/ComfyUI/tags?per_page=1" | grep -o '"name": "[^"]*' | head -1 | cut -d'"' -f4)

if [ -n "$LATEST_COMFYUI_VERSION" ]; then
    echo "Current ComfyUI version in build: $COMFYUI_VERSION"
    echo "Latest ComfyUI version available: $LATEST_COMFYUI_VERSION"
    if [ -t 0 ]; then
        read -p "Update to latest? [y/N]: " update
        if [[ "$update" =~ ^[Yy]$ ]]; then
            COMFYUI_VERSION="$LATEST_COMFYUI_VERSION"
        fi
    else
        echo "Non-interactive shell detected, keeping $COMFYUI_VERSION"
    fi
else
    echo "⚠️ Could not fetch latest ComfyUI tag, keeping $COMFYUI_VERSION"
fi

echo "📌 Baking ComfyUI version: $COMFYUI_VERSION"

# Check Docker login
echo "🔐 Checking Docker Hub login..."
if ! docker login 2>&1 | grep -q "Login Succeeded\|Already logged in"; then
    echo "❌ Not logged in to Docker Hub. Please run 'docker login' first."
    exit 1
fi
echo "✅ Docker Hub login confirmed"

# Setup buildx for cross-platform builds
echo "🔧 Using desktop-linux builder globally..."
docker buildx use --global desktop-linux

# Build the image with platform specification
# --pull removed: was invalidating all cache layers on every build
# Cache enabled: only changed layers are rebuilt (much faster)
echo "📦 Building image: ${IMAGE_NAME}:${TAG} for platform: ${PLATFORM}"
docker buildx build --builder desktop-linux --platform ${PLATFORM} --build-arg COMFYUI_VERSION="$COMFYUI_VERSION" --build-arg CACHEBUST=$(date +%s) -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG} --load .

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
docker push ${IMAGE_NAME}:${TAG}

echo "✅ Build and push completed!"
echo "📋 Image: ${IMAGE_NAME}:${TAG}"
echo "🌐 Available on Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
