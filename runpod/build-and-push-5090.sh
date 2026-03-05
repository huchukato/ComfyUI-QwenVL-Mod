#!/bin/bash

# Build and Push Script for ComfyUI-QwenVL-Mod (RTX 5090)
# CUDA 12.8 optimized for RTX 5090

set -e

echo "🐳 Building ComfyUI-QwenVL-Mod Docker image for RTX 5090..."

# Build variables
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="5090"
DOCKERFILE="Dockerfile"
PLATFORM="linux/x86_64"

# Check Docker login
echo "🔐 Checking Docker Hub login..."
if ! docker info | grep -q "Username"; then
    echo "❌ Not logged in to Docker Hub. Please run 'docker login' first."
    exit 1
fi
echo "✅ Docker Hub login confirmed"

# Setup buildx for cross-platform builds
echo "🔧 Using existing buildx for x86_64 platform..."
docker buildx use desktop-linux

# Build the image with platform specification
echo "📦 Building image: ${IMAGE_NAME}:${TAG} for platform: ${PLATFORM}"
docker buildx build --platform ${PLATFORM} -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG} --load .

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
docker push ${IMAGE_NAME}:${TAG}

echo "✅ Build and push completed!"
echo "📋 Image: ${IMAGE_NAME}:${TAG}"
echo "🌐 Available on Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
