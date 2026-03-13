# Build Scripts - Docker Images

## 📁 Script Files

### 🏗️ Local Build Scripts (Public)
For users who want to build and run ComfyUI locally:

- **`build-local-5090.sh`** - Build RTX 5090 image locally
- **`build-local-4090.sh`** - Build RTX 4090 image locally

**Usage:**
```bash
./build-local-5090.sh    # RTX 5090 (CUDA 12.8)
./build-local-4090.sh    # RTX 4090 (CUDA 12.4)
```

**Output:** Local Docker images (`comfyui-qwenvl-local:5090` / `comfyui-qwenvl-local:4090`)

### 🔐 Private Build Scripts (Maintainer Only)
For repository maintainer to build and push to Docker Hub:

- **`build-and-push-5090-private.sh`** - Build & push RTX 5090 to Docker Hub
- **`build-and-push-4090-private.sh`** - Build & push RTX 4090 to Docker Hub

**Usage:** (Maintainer only)
```bash
./build-and-push-5090-private.sh    # Push to huchukato/comfyui-qwenvl-runpod
./build-and-push-4090-private.sh    # Push to huchukato/comfyui-qwenvl-runpod-4090
```

## 🎯 Why This Separation?

### ✅ Benefits:
- **Security**: Only maintainer can push to Docker Hub
- **Clarity**: Users know exactly what scripts do
- **Functionality**: Everyone can use Docker locally
- **No Confusion**: Clear distinction between local vs production builds

### 🚀 For Users:
- Use `build-local-*.sh` for local Docker development
- Use pre-built images from Docker Hub for RunPod deployment

### 🔐 For Maintainer:
- Use `build-and-push-*-private.sh` for production builds
- Private scripts are NOT committed to public repository

## 📦 Docker Hub Images

### Production Images (Ready for RunPod):
- **RTX 5090**: `huchukato/comfyui-qwenvl-runpod:latest`
- **RTX 4090**: `huchukato/comfyui-qwenvl-runpod-4090:latest`

### Local Images (For Development):
- **RTX 5090**: `comfyui-qwenvl-local:5090`
- **RTX 4090**: `comfyui-qwenvl-local:4090`

## 🐳 Local Docker Run

After building locally:

```bash
# RTX 5090
docker run -d --gpus all \
  -p 8080:8080 -p 8888:8888 -p 8188:8188 \
  -v $(pwd)/output:/ComfyUI/output \
  comfyui-qwenvl-local:5090

# RTX 4090  
docker run -d --gpus all \
  -p 8080:8080 -p 8888:8888 -p 8188:8188 \
  -v $(pwd)/output:/ComfyUI/output \
  comfyui-qwenvl-local:4090
```

## 📋 Port Mappings

- **8080** → FileBrowser
- **8888** → JupyterLab (Terminal included)
- **8188** → ComfyUI
