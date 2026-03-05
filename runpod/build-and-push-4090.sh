#!/bin/bash

# Build and Push Script for ComfyUI-QwenVL-Mod (RTX 4090)
# CUDA 11.8 optimized for RTX 4090

set -e

echo "🐳 Building ComfyUI-QwenVL-Mod Docker image for RTX 4090..."

# Build variables
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="4090"
DOCKERFILE="Dockerfile.4090"

# Build the image
echo "📦 Building image: ${IMAGE_NAME}:${TAG}"
docker build -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG} .

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
docker push ${IMAGE_NAME}:${TAG}

# Also push as latest for 4090
echo "🏷️  Tagging and pushing as latest-4090..."
docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:latest-4090
docker push ${IMAGE_NAME}:latest-4090

echo "✅ Build and push completed!"
echo "📋 Image: ${IMAGE_NAME}:${TAG}"
echo "🌐 Available on Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
