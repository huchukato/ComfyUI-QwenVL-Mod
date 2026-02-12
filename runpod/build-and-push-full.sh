#!/bin/bash

# ComfyUI-QwenVL-Mod Full Build Script
# This script automates the process of building and pushing Docker images to Docker Hub
# for the ComfyUI-QwenVL-Mod project with ALL custom nodes included

set -e  # Exit on any error

# Configuration
IMAGE_NAME="huchukato/comfyui-qwenvl-runpod"
TAG="full"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
DATE_TAG="${IMAGE_NAME}:full-$(date +%Y%m%d)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker daemon is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker daemon is running"
}

# Function to check if user is logged in to Docker Hub
check_docker_login() {
    if ! docker login --get-token > /dev/null 2>&1; then
        print_warning "You are not logged in to Docker Hub."
        echo "Please run 'docker login' to authenticate."
        read -p "Do you want to continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Docker Hub authentication verified"
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building ComfyUI-QwenVL-Mod Full Docker image..."
    print_status "This may take 30-60 minutes depending on your internet connection and system performance."
    
    # Build using buildx for better performance and multi-platform support
    if docker buildx ls | grep -q "cloud-builder"; then
        print_status "Using existing cloud-builder instance"
        BUILDER="cloud-builder"
    else
        print_status "Creating new buildx builder instance"
        docker buildx create --name cloud-builder --use
        BUILDER="cloud-builder"
    fi
    
    # Build the image
    docker buildx build \
        --builder ${BUILDER} \
        --platform linux/amd64 \
        --tag "${FULL_IMAGE_NAME}" \
        --tag "${DATE_TAG}" \
        --push \
        -f Dockerfile.full \
        .
    
    print_success "Docker image built and pushed successfully!"
}

# Function to test the image locally
test_image() {
    print_status "Testing the Docker image locally..."
    
    # Pull the image to ensure it's available locally
    docker pull "${FULL_IMAGE_NAME}"
    
    # Run a quick test to verify the image starts correctly
    print_status "Running container test (this will start and stop quickly)..."
    if docker run --rm --gpus all "${FULL_IMAGE_NAME}" python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" > /dev/null 2>&1; then
        print_success "Image test passed!"
    else
        print_warning "Image test failed, but the build may still be successful"
    fi
}

# Function to display image information
display_image_info() {
    print_status "Image Information:"
    echo "  • Repository: ${IMAGE_NAME}"
    echo "  • Tags: full, full-$(date +%Y%m%d)"
    echo "  • Full image name: ${FULL_IMAGE_NAME}"
    echo "  • Date-tagged image: ${DATE_TAG}"
    echo ""
    
    # Get image size
    if docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -q "${IMAGE_NAME}"; then
        echo "  • Image size:"
        docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep "${IMAGE_NAME}" | while read line; do
            echo "    $line"
        done
    fi
    
    echo ""
    print_status "Docker Hub Repository: https://hub.docker.com/r/${IMAGE_NAME}"
}

# Function to display usage instructions
display_usage() {
    echo ""
    print_status "Usage Instructions:"
    echo ""
    echo "1. Pull the image:"
    echo "   docker pull ${FULL_IMAGE_NAME}"
    echo ""
    echo "2. Run the container:"
    echo "   docker run -d --gpus all -p 8188:8188 -p 8080:8080 -p 8888:8888 -p 22:22 ${FULL_IMAGE_NAME}"
    echo ""
    echo "3. Access services:"
    echo "   • ComfyUI: http://localhost:8188"
    echo "   • FileBrowser: http://localhost:8080"
    echo "   • JupyterLab: http://localhost:8888"
    echo "   • SSH: ssh root@localhost -p 22"
    echo ""
    echo "4. For RunPod deployment:"
    echo "   • Use image: ${FULL_IMAGE_NAME}"
    echo "   • GPU: RTX A4000 or higher recommended"
    echo "   • Storage: 100GB minimum"
    echo "   • Container disk: 50GB minimum"
    echo ""
}

# Main execution
main() {
    echo "========================================"
    echo "ComfyUI-QwenVL-Mod Full Build"
    echo "========================================"
    echo ""
    
    # Check prerequisites
    check_docker
    check_docker_login
    
    # Build the image
    build_image
    
    # Test the image
    test_image
    
    # Display information
    display_image_info
    display_usage
    
    print_success "Build process completed successfully!"
}

# Run the main function
main "$@"
