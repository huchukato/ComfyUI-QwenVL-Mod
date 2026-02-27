# ComfyUI-QwenVL-Mod - Enhanced Vision-Language with WAN 2.2

**Version 2.2.3** (2026/02/27) - 🧠 Advanced Memory Management + Documentation Updates

---

## 🌟 What is ComfyUI-QwenVL-Mod?

A powerful **enhanced vision-language node** for ComfyUI that combines Qwen3-VL models with professional WAN 2.2 video generation workflows. Features multilingual support, visual style detection, and NSFW capabilities for professional AI content creation.

**Think:** *"Your all-in-one solution for intelligent prompt enhancement and video generation with cutting-edge AI models!"*

---

## 🎬 Key Features

### **🚀 WAN 2.2 Video Generation**
- **Text-to-Video (T2V)**: Professional 5-second video generation
- **Image-to-Video (I2V)**: Advanced image animation with style detection
- **Story Generation**: 20-second continuous videos with 4 narrative segments
- **Storyboard Workflows**: Seamless storyboard-to-storyboard generation
- **Cinematic Video**: Professional cinematography specifications

### **🌐 Enhanced Capabilities**
- **Multilingual Support**: Process prompts from any language (Italian, English, etc.)
- **Visual Style Detection**: 12+ artistic styles (anime, 3D, pixel art, puppet animation, etc.)
- **Smart Prompt Caching**: Performance optimization with Fixed Seed Mode
- **GGUF Backend**: Efficient local model inference with quantization support
- **NSFW Support**: Comprehensive content generation without restrictions

### **🧠 Intelligent Features**
- **Auto-Prompt Enhancement**: Automatically enhance user prompts for optimal generation
- **Professional Cinematography**: Built-in specifications for lighting, camera angles, shot types
- **Timeline Structure**: Precise 5-second timeline with frame-by-frame descriptions
- **Memory Management**: Advanced `unload_after_run` parameter for aggressive memory cleanup on VRAM-limited systems
- **Keep Last Prompt**: Generate once, preserve results while changing inputs

---

## 🎯 What's New in v2.2.3
- **Memory Management**: Added `unload_after_run` parameter for aggressive VRAM cleanup
- **Documentation**: Updated all README files with new memory features
- **Credits**: Added GenorTG credits for memory management innovations

## 🎯 What's New in v2.2.2

### **🚀 Critical T2V/I2V Workflow Fixes**
- **Batch Processing**: Fixed critical T2V → GGUF issue with batch images
- **Frame Detection**: Added automatic batch detection and individual frame processing
- **Video Support**: Enhanced video frame processing with proper shape handling
- **Debug Enhanced**: Comprehensive logging for batch processing troubleshooting

### **🔄 Same Model Reuse Fix**
- **Conflict Resolution**: Fixed crash when using same model between T2V and I2V nodes
- **Memory Management**: Enhanced cleanup with CUDA synchronization and timing
- **Signature Mismatch**: Resolved different signature patterns between nodes
- **Aggressive Cleanup**: Forced complete VRAM cleanup before model reload

### **🔧 keep_model_loaded Enhancement**
- **Missing Parameter**: Added keep_model_loaded to PromptEnhancer node
- **Consistent Behavior**: Both GGUF and PromptEnhancer now have identical memory management
- **Conditional Cleanup**: Proper cleanup based on keep_model_loaded setting
- **User Control**: Full control over memory usage vs performance

---

## 🎬 WAN 2.2 Story Workflow - Revolutionary AI Storytelling

### **📖 AI Story Generation**
- **4-Segment Videos**: Automatic 20-second videos (4 × 5-second segments)
- **Narrative Continuity**: Perfect story flow between segments
- **NSFW Support**: Enhanced adult content generation
- **Timeline-Free**: Natural storytelling without time markers

### **🔄 Smart Auto-Split**
- **Story Split Node**: Intelligent prompt separation technology
- **Auto-Detection**: Handles any separator format automatically
- **4-Output Guarantee**: Always produces exactly 4 prompts
- **Debug Mode**: Built-in troubleshooting information

---

## 📦 Installation

### **Requirements**
- **ComfyUI**: v0.13.0+
- **GPU**: 8GB+ VRAM (16GB+ recommended)
- **System**: Windows/Linux/Mac
- **Python**: 3.10+ (or use provided Docker environment)

### **Quick Install**
1. Download ComfyUI-QwenVL-Mod v2.2.0
2. Extract to `ComfyUI/custom_nodes/ComfyUI-QwenVL-Mod`
3. Restart ComfyUI
4. Load included workflows

### **Docker/Cloud Ready**
- **RunPod**: Pre-configured templates available
- **VastAI**: Optimized instances ready
- **Local**: Docker support included

---

## 🎮 Usage Examples

### **Basic Text-to-Video**
1. Load **WAN2.2-I2V-AutoPrompt.json**
2. Input your text prompt
3. Select model (HF or GGUF)
4. Generate enhanced video

### **Image-to-Video with Style**
1. Load **WAN2.2-I2V-AutoPrompt.json**
2. Upload your image
3. Enable style detection
4. Generate animated video

### **AI Story Generation**
1. Load **WAN2.2-I2V-AutoPrompt-Story.json**
2. Input your story idea
3. Auto-split into 4 segments
4. Generate 20-second story video

---

## 🔧 Technical Specifications

### **⚡ Performance**
- **Context**: 65,536 tokens (8B models)
- **Memory**: Optimized VRAM usage
- **Stability**: Crash-free operation
- **Speed**: Fast generation times

### **🎨 Model Support**
- **Qwen3-VL 4B**: 7 GGUF variants (2.38GB-4.28GB)
- **Qwen3-VL 8B**: 7 GGUF variants (4.8GB-8.71GB)
- **HF Models**: Josiefed and official variants
- **Quantization**: Q4_K_S, Q5_K_S for VRAM efficiency

### **🌐 Multilingual Capabilities**
- **Input Languages**: Any language supported
- **Auto-Translation**: Automatic translation to optimized English
- **Style Detection**: Works with multilingual prompts
- **Cultural Adaptation**: Context-aware prompt enhancement

---

## 🎯 Included Workflows

### **🍿 WAN 2.2 Presets**
- **Wan 2.2 I2V**: Image-to-video with timeline structure
- **Wan 2.2 T2V**: Text-to-video with professional specs
- **Wan Extended Storyboard**: Multi-segment continuity
- **Wan Cinematic Video**: Single scene with cinematography

### **🔥 Advanced Features**
- **NSFW Enhancement**: Uncensored content generation
- **Professional Lighting**: 8 light types + 9 qualities
- **Camera Control**: 6 shot types + 5 compositions
- **Color Grading**: 4 tone options

---

## 🎨 Visual Style Detection

Automatically detects and enhances:
- **Anime style** - Japanese animation aesthetics
- **3D cartoon** - Computer-generated animation
- **Pixel art** - Retro gaming graphics
- **Puppet animation** - Stop-motion style
- **3D game style** - Video game graphics
- **Claymation** - Clay animation
- **Watercolor** - Painting style
- **Black and white animation** - Monochrome
- **Oil painting style** - Classical art
- **Felt style** - Textile art
- **Tilt-shift photography** - Miniature effect
- **Time-lapse photography** - Speed photography

---

## 🔥 NSFW Content Support

### **Enhanced Generation**
- **Explicit Content**: Uncensored adult descriptions
- **Detailed Scenes**: 8-12 sentences per segment
- **Natural Progression**: Smooth story flow
- **Style Adaptation**: Automatic visual style matching
- **Quality**: Consistent characters & scenes

### **Professional Applications**
- **Adult Content**: Industry-standard generation
- **Artistic Nudity**: Classical art styles
- **Educational**: Anatomy and artistic study
- **Creative**: Artistic expression

---

## 🚀 Why Choose ComfyUI-QwenVL-Mod?

### **🎬 For Content Creators**
- **Storytelling**: Create compelling narratives
- **Efficiency**: One prompt → complete video
- **Quality**: Professional video output
- **Flexibility**: Any genre, any style

### **🔥 For NSFW Content**
- **Explicit**: Uncensored generation
- **Detailed**: Rich scene descriptions
- **Continuous**: Smooth story flow
- **Natural**: Realistic progression

### **⚡ For Power Users**
- **Customizable**: Easy to modify
- **Extendable**: Add more segments
- **Integrable**: Works with existing setups
- **Optimized**: Maximum performance

---

## 📥 Download & Support

### **Get Started**
1. **Download**: ComfyUI-QwenVL-Mod v2.2.0
2. **Install**: Follow standard installation
3. **Load**: Included workflows
4. **Create**: Your first AI-enhanced video!

### **Community & Support**
- **GitHub**: [Repository](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
- **Issues**: Report bugs and request features
- **Discord**: Community support (coming soon)
- **Documentation**: Complete guides and tutorials

---

## 🌟 What Makes This Special?

- **First**: Complete AI story system with vision enhancement
- **Smart**: Intelligent prompt splitting and enhancement
- **Complete**: End-to-end solution from text to video
- **Optimized**: Performance-tuned for professional use
- **Ready**: Works out-of-the-box with included workflows

---

## 🎬 Create Amazing AI Videos Today!

Transform your ideas into stunning videos with the power of Qwen3-VL vision enhancement and WAN 2.2 video generation.

**Perfect for creators, artists, and professionals looking for the ultimate AI video enhancement tool!** 🌟

---

**Built with ❤️ for the ComfyUI community**
