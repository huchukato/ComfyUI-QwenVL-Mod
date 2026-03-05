#!/bin/bash

# Build and Push Script for ComfyUI-QwenVL-Mod (RTX 5090)
# CUDA 12.8 optimized for RTX 5090

set -e

echo "🐳 Building ComfyUI-QwenVL-Mod Docker image for RTX 5090..."

# Build variables
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="5090"
DOCKERFILE="Dockerfile"

# Build the image
echo "📦 Building image: ${IMAGE_NAME}:${TAG}"
docker build -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG} .

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
docker push ${IMAGE_NAME}:${TAG}

# Also push as latest for 5090
echo "🏷️  Tagging and pushing as latest-5090..."
docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:latest-5090
docker push ${IMAGE_NAME}:latest-5090

echo "✅ Build and push completed!"
echo "📋 Image: ${IMAGE_NAME}:${TAG}"
echo "🌐 Available on Docker Hub: https://hub.docker.com/r/${IMAGE_NAME}"
