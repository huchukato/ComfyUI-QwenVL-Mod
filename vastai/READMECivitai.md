# ComfyUI-QwenVL-Mod - Enhanced Vision-Language with WAN 2.2

**Version 2.2.4** (2026/03/13) - 🎬 Critical I2V Timeline Fixes & NSFW Presets Optimization

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
- **Keep Last Prompt**: Generate once, preserve results while changing inputs

---

## 🎯 What's New in v2.2.4 - CRITICAL I2V TIMELINE FIXES

### **🚨 Major I2V Timeline (20s) Fixes**
- **✅ Style Coherence**: Fixed AI changing anime→realism mid-sequence
- **✅ Character Stability**: Fixed characters disappearing/appearing incorrectly  
- **✅ Natural Lighting**: Fixed AI adding artificial lights not in image
- **✅ Timeline Structure**: Fixed continuous numbering (6,7,8...) instead of 0-5 restart
- **✅ Format Consistency**: Fixed missing parentheses and unwanted labels
- **✅ Output Format**: Each prompt starts directly with timeline markers

### **🔧 NSFW Presets Optimization**
- **✅ Complete Specifications**: All 8 NSFW presets now include full NSFW descriptions
- **✅ Emoji Display**: Restored proper emoji rendering (🍿🎥🎬📖)
- **✅ Clear Instructions**: Removed confusing recommendations from presets
- **✅ User Guide**: Token settings guide created for workflow optimization

### **📋 Technical Improvements**
- **✅ Timeline Markers**: Correct `(At X seconds: ...)` format for all 4 prompts
- **✅ Character Continuity**: Natural progression without forced artificial presence
- **✅ Lighting Rules**: Logical progression instead of absolute prohibitions
- **✅ Style Detection**: Consistent style application across all timeline segments

### **🎯 Model Recommendations**
- **Qwen3-VL-8B**: Recommended for I2V Timeline (20s) complex sequences
- **Qwen3-VL-4B**: Sufficient for I2V Scene (5s) single prompts
- **Token Settings**: 2048+ for 20s timeline, 1024+ for 5s prompts

## 🎯 What's New in v2.2.3
- **CUDA 13 Compatibility**: Fixed crashes caused by conflicting unload operations
- **Parameter Cleanup**: Removed redundant unload_after_run from all nodes
- **Bug Fixes**: Resolved "missing required positional argument" errors
- **Memory Management**: Streamlined VRAM cleanup with VRAM Cleanup node
- **Documentation**: Updated all README files with new memory features
- **Credits**: Added community credits for feedback and testing

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

## 🚨 CRITICAL BUG FIXES - v2.2.4

### **🎬 I2V Timeline (20s) - COMPLETELY FIXED**
**Before v2.2.4:**
- ❌ Anime style changed to realism mid-sequence
- ❌ Characters disappeared/appeared randomly
- ❌ AI added artificial lights not in image
- ❌ Timeline numbering: 6,7,8... instead of 0-5 restart
- ❌ Missing parentheses and unwanted "Prompt 1:" labels

**After v2.2.4:**
- ✅ **Perfect Style Coherence**: Anime stays anime, realism stays realism
- ✅ **Character Stability**: Same characters throughout all 4 prompts
- ✅ **Natural Lighting**: Only lights visible in image, logical progression
- ✅ **Correct Timeline**: Each prompt uses 0-5 seconds format
- ✅ **Clean Output**: Proper `(At X seconds: ...)` format, no labels

### **🔥 NSFW Presets - ENHANCED & FIXED**
- ✅ **Complete Specifications**: All 8 presets with full explicit descriptions
- ✅ **Emoji Display**: Proper 🍿🎥🎬📖 icons (no more unicode codes)
- ✅ **User-Friendly**: Removed confusing technical recommendations
- ✅ **Token Guide**: Workflow note for optimal settings

**🎯 Result**: Perfect I2V Timeline generation every time!

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

### **Docker/Cloud Ready**
- **RunPod**: Pre-configured templates available
- **VastAI**: Optimized instances ready
- **Local**: Docker support included

### **Quick Install**
1. **Download**: ComfyUI-QwenVL-Mod (latest version)
2. **Extract** to `ComfyUI/custom_nodes/ComfyUI-QwenVL-Mod`
3. **Restart** ComfyUI
4. **Load** included workflows

---

## 🎮 Usage Examples

### **Basic Image-to-Video**
1. Load **WAN2.2-I2V-AutoPrompt.json**
2. Upload your image
3. Select model (HF or GGUF)
4. Generate enhanced video

### **Basic Text-to-Video**
1. Load **WAN2.2-T2V-AutoPrompt.json**
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
