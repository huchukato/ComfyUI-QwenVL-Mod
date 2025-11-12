# **QwenVL for ComfyUI**

ComfyUI-QwenVL 是一款自定义节点，它集成了来自阿里云的强大 Qwen-VL 系列视觉语言模型（LVLMs），包括最新的 Qwen3-VL 和 Qwen2.5-VL。这款高级节点能够在您的 ComfyUI 工作流中实现无缝的多模态 AI 功能，支持高效的文本生成、图像理解和视频分析。

![QwenVL_V1.1.0](https://github.com/user-attachments/assets/13e89746-a04e-41a3-9026-7079b29e149c)

## **📰 新闻与更新**

* **2025/10/17**: **v1.0.0** 初始版本发布
[![QwenVL_V1.0.0r](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/example_workflows/QWenVL.jpg)](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/example_workflows/QWenVL.json)
  * 支持 Qwen3-VL 和 Qwen2.5-VL 系列模型。  
  * 自动从 Hugging Face 下载模型。  
  * 支持即时量化（4-bit、8-bit、FP16）。  
  * 提供预设和自定义提示词系统，使用灵活方便。  
  * **包含**一个标准节点和一个高级**节点**，满足不同层次用户的需求。  
  * 具备硬件感知保护机制，以兼容 FP8 模型。  
  * 支持图像和视频（帧序列）输入。  
  * 提供“保持模型加载”选项，以提高连续运行的性能。  
  * **包含种子（Seed）参数**，用于生成可复现的结果。

## **✨ 功能特性**

* **标准与高级节点**：包含一个用于快速上手的简单 QwenVL 节点，以及一个提供精细生成控制的 QwenVL (Advanced) 节点。  
* **预设与自定义提示词**：可从一系列便捷的预设提示词中选择，或自行编写以实现完全控制。  
* **多模型支持**：轻松在各种官方 Qwen-VL 模型之间切换。  
* **自动模型下载**：首次使用时会自动下载所需模型。  
* **智能量化**：通过 4-bit、8-bit 和 FP16 选项，平衡显存占用与性能。  
* **硬件感知**：自动检测 GPU 能力，并防止因模型不兼容（例如 FP8）而导致的错误。  
* **可复现生成**：使用 seed 参数可获得一致的输出结果。  
* **内存管理**：“保持模型加载”选项可将模型保留在显存中，以加快处理速度。  
* **图像与视频支持**：接受单个图像和视频帧序列作为输入。  
* **强大的错误处理**：为硬件或内存问题提供清晰的错误信息。  
* **简洁的控制台输出**：在操作过程中提供最少且信息丰富的控制台日志。
* **新增 attention_mode 参数，支持自动启用 Flash-Attention v2 加速。
* **新增 use_torch_compile 参数，在 Torch 2.1+ 上可启用 JIT 优化，提高推理吞吐。
* **新增 device 参数，支持在 CUDA、CPU 与 Apple MPS 之间手动切换。
* **改进显存检测逻辑，可在内存不足时自动降低量化等级。

## **🚀 安装**

1. 将此仓库克隆到您的 ComfyUI/custom\_nodes 目录：  
```
   cd ComfyUI/custom\_nodes  
   git clone https://github.com/1038lab/ComfyUI-QwenVL.git\
```
2. 安装所需的依赖项：  
```
   cd ComfyUI/custom\_nodes/ComfyUI-QwenVL  
   pip install \-r requirements.txt
```
3. 重启 ComfyUI。

## **📥 下载模型**

模型将在首次使用时自动下载。如果您希望手动下载，请将它们放置在 ComfyUI/models/LLM/Qwen-VL/ 目录下。

| 模型 | 链接 |
| :---- | :---- |
| Qwen3-VL-2B-Instruct | [下载](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| Qwen3-VL-2B-Thinking | [下载](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) |
| Qwen3-VL-2B-Instruct-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8) |
| Qwen3-VL-2B-Thinking-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8) |
| Qwen3-VL-4B-Instruct | [下载](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen3-VL-4B-Thinking | [下载](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) |
| Qwen3-VL-4B-Instruct-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8) |
| Qwen3-VL-4B-Thinking-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-FP8) |
| Qwen3-VL-8B-Instruct | [下载](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-8B-Thinking | [下载](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) |
| Qwen3-VL-8B-Instruct-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) |
| Qwen3-VL-8B-Thinking-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-FP8) |
| Qwen3-VL-32B-Instruct | [下载](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL-32B-Thinking | [下载](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking) |
| Qwen3-VL-32B-Instruct-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) |
| Qwen3-VL-32B-Thinking-FP8 | [下载](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8) |
| Qwen2.5-VL-3B-Instruct | [下载](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL-7B-Instruct | [下载](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |


## **📖 基本用法**

1. 从 🧪AILab/QwenVL 类别中添加 **"QwenVL"** 节点。  
2. 选择您希望使用的 model\_name（模型名称）。  
3. 连接一个图像或视频（图像序列）源到该节点。  
4. 编写您的提示词。  
5. 根据需要调整其他参数并运行工作流。

### **⚙️ 参数详解**

| 参数 | 描述 | 默认值 |
| :---- | :---- | :---- |
| **model\_name** | 要使用的 Qwen-VL 模型。 | Qwen3-VL-4B-Instruct |
| **quantization** | 即时量化级别。对于预量化模型（如 FP8）将被忽略。 | 8-bit (Balanced) |
| **attention_mode** |	注意力机制模式。auto 会自动启用 Flash-Attn v2（若可用），否则回退为 SDPA。|	auto	auto、flash_attention_2、sdpa |	标准 & 高级 |
| **use_torch_compile** |	启用 torch.compile('reduce-overhead') 优化以提升 CUDA 推理性能（需 Torch 2.1+）。	| False | - | 高级 |
| **preset\_prompt** | 为常见任务预定义的一系列提示词。 | "Describe this..." |
| **custom\_prompt** | 自定义文本提示词。如果提供，将覆盖预设提示词。 |  |
| **max\_tokens** | 要生成的最大新词元（token）数量。 | 1024 |
| **temperature** | 控制随机性。值越高 \= 更具创造性。（当 num\_beams 为 1 时使用）。 | 0.6 |
| **top\_p** | 核心采样阈值。（当 num\_beams 为 1 时使用）。 | 0.9 |
| **num\_beams** | 用于束搜索（beam search）的光束数量。\> 1 时将禁用 temperature/top\_p 采样。 | 1 |
| **repetition\_penalty** | 抑制重复词元的惩罚系数。1.0 表示中性。 | 1.2 |
| **keep\_model\_loaded** | 将模型保留在显存中，以便后续运行更快。 | True |
| **seed** | 随机种子，用于确保生成结果的可复现性。 | 1 |

### **💡 量化选项**

| 模式 | 精度 | 显存占用 | 速度 | 质量 | 推荐适用场景 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| None (FP16) | 16位浮点 | 高 | 最快 | 最佳 | 高显存 GPU (16GB+) |
| 8-bit (Balanced) | 8位整数 | 中 | 较快 | 很好 | 追求均衡性能 (8GB+) |
| 4-bit (VRAM-friendly) | 4位整数 | 低 | 较慢\* | 好 | 低显存 GPU (\<8GB) |

**\*关于 4-bit 速度的说明**：4-bit 量化能显著减少显存使用，但由于实时反量化的计算开销，在某些系统上可能会导致性能下降。

### **🤔 设置技巧**

| 设置 | 建议 |
| :---- | :---- |
| **模型选择** | 对于大多数用户，Qwen3-VL-4B-Instruct 是一个很好的起点。如果您有 40 系 GPU，可以尝试 \-FP8 版本以获得更好的性能。 |
| **内存模式** | 如果您计划多次运行该节点，请保持 keep\_model\_loaded 启用（True）以获得最佳性能。仅在其他节点需要更多显存时才禁用它。 |
| **量化** | 从默认的 8-bit 开始。如果您的显存充裕（\>16GB），切换到 None (FP16) 以获得最佳速度和质量。如果显存不足，请使用 4-bit。 |
| **性能** | 首次加载具有特定量化设置的模型时可能会较慢。后续的运行（在启用 keep\_model\_loaded 的情况下）会快得多。 |

## **🧠 关于模型**

此节点利用了由阿里云 Qwen 团队开发的 Qwen-VL 系列模型。这些是功能强大的开源大型视觉语言模型（LVLMs），旨在理解和处理视觉及文本信息，非常适合用于详细的图像和视频描述等任务。

## **🗺️ 路线图**

### **✅ 已完成 (v1.0.0)**

* ✅ 支持 Qwen3-VL 和 Qwen2.5-VL 模型。  
* ✅ 自动模型下载和管理。  
* ✅ 即时 4-bit、8-bit 和 FP16 量化。  
* ✅ 针对 FP8 模型的硬件兼容性检查。  
* ✅ 支持图像和视频（帧序列）输入。  
* ✅ **即将支持 GGUF 模型格式。**

### **🔄 未来计划**

* GGUF 格式支持 CPU 和更广泛的硬件兼容性。  
* 集成更多视觉语言模型。  
* 提供更高级的参数选项以微调生成过程。  
* 支持额外的视频处理功能。

## **🙏 致谢**

* **Qwen 团队**：[阿里云](https://github.com/QwenLM) \- 感谢其开发并开源了强大的 Qwen-VL 模型。  
* **ComfyUI**：[comfyanonymous](https://github.com/comfyanonymous/ComfyUI) \- 感谢其创造了如此出色且可扩展的 ComfyUI 平台。  
* **ComfyUI 集成**：[1038lab](https://github.com/1038lab) \- 本自定义节点的开发者。

## **📜 许可证**

此仓库的代码根据 [GPL-3.0 许可证](LICENSE) 发布。
