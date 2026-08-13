#!/bin/bash

# Build and Push Script for ComfyUI-QwenVL-Mod (RunPod, CUDA 13.0, LTX 2.3 edition)
# Based on runpod/comfyui:cuda13.0 template

set -e

echo "🐳 Building ComfyUI-QwenVL-Mod Docker image (RunPod, CUDA 13.0, LTX 2.3)..."

# Build variables
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="cu13-ltx"
DOCKERFILE="Dockerfile.CU13-LTX"
PLATFORM="linux/amd64"

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
docker buildx build --builder desktop-linux --platform ${PLATFORM} --build-arg CACHEBUST=$(date +%s) -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG} --load .

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
docker push ${IMAGE_NAME}:${TAG}

echo "✅ Build and push completed!"
echo "📋 Image: ${IMAGE_NAME}:${TAG}"
echo "🌐 Available on Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
