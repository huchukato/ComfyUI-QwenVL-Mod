# 🧩 ComfyUI-QwenVL — Custom Models Guide

You can add your own HuggingFace or local fine-tuned QwenVL models  
without modifying the main code.

---

## 📁 File location
Place a file named `custom_models.json` in this folder:

```

ComfyUI/custom_nodes/ComfyUI-QwenVL/

````

When ComfyUI starts, it will automatically detect and merge this file  
with the default `config.json`.

---

## ⚙️ File format
Use the following JSON structure:

```json
{
  "hf_models": {
    "My QwenVL Finetune": {
      "repo_id": "myusername/MyQwenVL-Finetune",
      "vram_requirement": {
        "4bit": 4,
        "8bit": 6,
        "full": 10
      }
    },
    "Tiny QwenVL": {
      "repo_id": "huggingface-user/qwenvl-tiny",
      "vram_requirement": {
        "4bit": 2,
        "8bit": 3,
        "full": 5
      }
    }
  }
}
````

* The **key** (e.g. `"My QwenVL Finetune"`) is what appears in the model dropdown list.
* `"repo_id"` can be a HuggingFace model name or a local directory path.
* `"vram_requirement"` is optional but helps ComfyUI display VRAM recommendations.

---

## 🧠 Notes

* If `custom_models.json` doesn’t exist, the system just skips it (no errors).
* You can edit and reload it anytime — simply restart ComfyUI to apply changes.
* Compatible with both Qwen2-VL and Qwen3-VL model architectures.
