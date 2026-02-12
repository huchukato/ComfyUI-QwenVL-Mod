# **QwenVL-Mod per ComfyUI**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue?style=for-the-badge&logo=python)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-GPL--3.0-green?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.9-orange?style=for-the-badge)](https://github.com/huchukato/ComfyUI-QwenVL-Mod/releases)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-black?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![HuggingFace](https://img.shields.io/badge/Models-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/Qwen)
[![Downloads](https://img.shields.io/github/downloads/huchukato/ComfyUI-QwenVL-Mod/total?style=for-the-badge&logo=github)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![Stars](https://img.shields.io/github/stars/huchukato/ComfyUI-QwenVL-Mod?style=for-the-badge&logo=github)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![Issues](https://img.shields.io/github/issues/huchukato/ComfyUI-QwenVL-Mod?style=for-the-badge&logo=github)](https://github.com/huchukato/ComfyUI-QwenVL-Mod/issues)

Il custom node ComfyUI-QwenVL integra la potente serie di modelli vision-linguaggio (LVLM) Qwen-VL di Alibaba Cloud, inclusi i più recenti Qwen3-VL e Qwen2.5-VL, oltre a backend GGUF e supporto solo testo Qwen3. Questo nodo avanzato abilita funzionalità multimodali AI senza soluzione di continuità nei tuoi workflow ComfyUI, permettendo efficiente generazione di testo, comprensione di immagini e analisi video.

<img width="749" height="513" alt="Qwen3-VL-Mod" src="https://github.com/user-attachments/assets/0f10b887-1953-4923-b813-37ccacb8a9aa" />

## **📰 Notizie & Aggiornamenti**
* **2026/02/12**: **v2.0.9** Parametro Bypass Mode per persistenza prompt. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-209-20260212)]
> [!NOTE]  
> 🎛️ **Bypass Mode**: Nuovo parametro `bypass_mode` permette di mantenere i prompt generati in precedenza senza rigenerarli.  
> 🔄 **Smart Cache**: Quando bypass mode è abilitato, i nodi recuperano il prompt cached più recente per il modello corrente.  
> 🎯 **Workflow Perfetto**: Genera prompt una volta, poi abilita bypass mode per preservarli cambiando gli input.  
> ⚡ **Zero Risorse**: Bypass mode non consuma risorse computazionali - risposta istantanea.  
> 📋 **Feature Universale**: Disponibile in tutti i nodi (HF, GGUF, PromptEnhancer, varianti Advanced).  
> 🎮 **Controllo Semplice**: Basta attivare/disattivare il checkbox bypass_mode per abilitare/disabilitare persistenza prompt.

* **2026/02/06**: **v2.0.8** Correzioni bug e miglioramenti stabilità. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-208-20260206)]
> [!NOTE]  
> 🐛 **Correzioni Bug**: Risolto errore sintassi JSON nei prompt di sistema e problemi variabili non definite.  
> 🌐 **Supporto Multilingua**: Supporto multilingua completo aggiunto a tutti i preset WAN 2.2.  
> 🎨 **Rilevamento Stile**: Migliorato rilevamento stile visivo per anime, 3D, pixel art e altro.  
> 🔧 **Stabilità**: Ripristinati miglioramenti seed fissi problematici per mantenere operazione stabile.  
> 📝 **Documentazione**: Aggiornato README e changelog con i miglioramenti di oggi.

* **2026/02/04**: **v2.0.7** Sistema intelligente cache prompt con Modalità Seed Fisso. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-207-20260204)]
> [!NOTE]  
> 🧠 **Cache Intelligente**: Cache automatica prompt previene rigenerazione di prompt identici.  
> 🔒 **Modalità Seed Fisso**: Imposta qualsiasi valore seed fisso per mantenere prompt consistenti indipendentemente dalle variazioni multimediali.  
> ⚡ **Boost Performance**: Risposta istantanea per prompt in cache con tempo zero di caricamento modello.  
> 🔧 **Manutenzione Codice**: Rimossi parametri deprecati da tutte le funzioni download per compatibilità futura.  
> 📈 **Miglioramento GGUF**: Aumentata dimensione contesto predefinita da 8192 a 32768 per migliore utilizzo modello.  
> 🔄 **Cache Universale**: Modalità Seed Fisso e cache intelligente ora disponibili su tutti i nodi (HF, GGUF, PromptEnhancer).  
> 🎯 **Logica Semplificata**: Sistema cache semplificato che include sempre seed per comportamento prevedibile su tutti i valori seed.

* **2026/02/03**: **v2.0.6** Miglioramento cinematografia professionale per tutti i preset WAN 2.2. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-206-20260203)]
> [!NOTE]  
> 🎬 **Specifiche Professionali**: Tutti i preset WAN 2.2 ora includono specifiche cinematografiche complete.  
> 📹 **Dettagli Tecnici**: Sorgenti luminose, tipi ripresa, specifiche lenti, movimenti camera, requisiti tonalità colore.  
> 🎯 **Branding Coerente**: Aggiornati nomi preset con branding famiglia WAN per migliore organizzazione.

* **2026/02/01**: **v2.0.5** Preset storyboard esteso aggiunto per continuità formato WAN 2.2. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-205-20260201)]
> [!NOTE]  
> 🎬 **Storyboard Esteso**: Nuovo preset per generazione storyboard-to-storyboard senza soluzione di continuità con formato timeline.  
> 🔄 **Focus Continuità**: Ogni paragrafo ripete contenuto precedente per transizioni fluide.  
> 🎯 **Compatibile WAN 2.2**: Stessa struttura timeline e supporto NSFW del preset I2V.

* **2026/02/01**: **v2.0.4** Aggiornamento stabilità - rimosso SageAttention per migliore compatibilità e affidabilità output modello. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-204-20260201)]
> [!NOTE]  
> 🔧 **SageAttention Rimosso**: Eliminata complessità e problemi interferenza per performance modello stabili.  
> ⚡ **Flash Attention 2**: Ancora disponibile per 2-3x velocità su hardware compatibile.  
> 🛡️ **Stabilità Migliorata**: Pipeline attention pulita con SDPA come fallback affidabile.

* **2026/02/01**: **v2.0.3** Correzione compatibilità SageAttention per patching corretto su versioni transformer. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-203-20260201)]
> [!NOTE]  
> 🔧 **Correzione Critica**: Risolto AttributeError che impediva SageAttention di funzionare con certe versioni transformer.  
> ⚡ **Performance Ripristinata**: 2-5x velocità ora funziona correttamente con quantizzazione 8-bit su hardware compatibile.

* **2026/02/01**: **v2.0.2** Accessibilità modelli migliorata, logica prompt personalizzata migliorata, espanso generazione contenuti NSFW. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-202-20260201)]
> [!NOTE]  
> 🚀 **Modelli Abliterated Free**: Aggiunti modelli senza token non censurati come predefiniti per migliore accessibilità.  
> 🔧 **Correzione Prompt Personalizzato**: Ora combina con template preset invece di sostituirli su tutti i nodi.  
> 📝 **NSFW Migliorato**: Descrizioni complete per contenuti adulti con specifiche dettagliate atti.  
> 🎬 **Priorità WAN 2.2**: Spostato preset generazione video in posizione superiore per accesso workflow più rapido.

* **2026/01/30**: **v2.0.1-enhanced** Aggiunto supporto SageAttention e integrazione WAN 2.2. [[Aggiornamenti](https://github.com/huchukato/ComfyUI-QwenVL-Mod/blob/main/update.md#version-201-enhanced-20260130)]
> [!NOTE]  
> 🚀 **SageAttention**: 2-5x boost performance con attention quantizzato 8-bit per GPU RTX 30+.  
> 🎬 **Integrazione WAN 2.2**: Nuovi prompt specializzati per generazione video cinematografica - converti immagini/video in descrizioni timeline 5-secondi (I2V) o testo a video (T2V) con direzione scena professionale.

* **2025/12/22**: **v2.0.0** Aggiunti nodi supportati GGUF e nodi Prompt Enhancer. [[Aggiornamenti](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/update.md#version-200-20251222)]
> [!IMPORTANT]  
> Installa llama-cpp-python prima di eseguire nodi GGUF [istruzioni](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md)

![600346260_122188475918461193_3763807942053883496_n](https://github.com/user-attachments/assets/bc9450d9-1695-452d-9e46-f05a4bf315de)
* **2025/11/10**: **v1.1.0** Revisione runtime con selettore modalità attention, rilevamento automatico flash-attn, cache più intelligente, e controlli quantizzazione/torch.compile in entrambi i nodi. [[Aggiornamenti](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/update.md#version-110-20251110)]
* **2025/10/31**: **v1.0.4** Modelli Personalizzati Supportati [[Aggiornamenti](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/update.md#version-104-20251031)]
* **2025/10/22**: **v1.0.3** Elenco modelli aggiornato [[Aggiornamenti](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/update.md#version-103-20251022)]
* **2025/10/17**: **v1.0.0** Rilascio Iniziale  
  * Supporto per serie modelli Qwen3-VL e Qwen2.5-VL.  
  * Download automatico modelli da Hugging Face.  
  * Quantizzazione al volo (4-bit, 8-bit, FP16).  
  * Sistema Preset e Prompt Personalizzati per uso flessibile e facile.  
  * **Include sia nodo standard che avanzato** per utenti di tutti i livelli.  
  * Salvaguardie hardware-aware per compatibilità modelli FP8.  
  * Supporto input immagine e video (sequenza frame).  
  * Opzione "Keep Model Loaded" per performance migliorate su esecuzioni sequenziali.  
  * **Parametro Seed** per generazione riproducibile.

[![QwenVL_V1.0.0r](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/example_workflows/QWenVL.jpg)](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/example_workflows/QWenVL.json)

## **✨ Funzionalità**

[![Multimodal](https://img.shields.io/badge/Multimodal-Image%20%7C%20Video%20%7C%20Text-purple?style=flat-square)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![Models](https://img.shields.io/badge/Models-Qwen3%20%7C%20Qwen2.5%20%7C%20GGUF-blue?style=flat-square)](https://huggingface.co/Qwen)
[![Quantization](https://img.shields.io/badge/Quantization-4%20%7C%208%20%7C%2016%20bit-orange?style=flat-square)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![Performance](https://img.shields.io/badge/Performance-Flash%20Attention%20%7C%20SDPA-green?style=flat-square)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![WAN2.2](https://img.shields.io/badge/WAN%202.2-Video%20Generation-red?style=flat-square)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![Caching](https://img.shields.io/badge/Caching-Smart%20%7C%20Persistent-yellow?style=flat-square)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![Bypass](https://img.shields.io/badge/Bypass-Prompt%20Persistence-green?style=flat-square)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)

* **Nodi Standard & Avanzati**: Include nodo QwenVL semplice per uso rapido e nodo QwenVL (Avanzato) con controllo fine-grained sulla generazione.  
* **Enhancer Prompt**: Enhancer prompt solo testo dedicati per backend HF e GGUF.  
* **Preset & Prompt Personalizzati**: Scegli da elenco comodo preset prompt o scrivi i tuoi per controllo completo. I prompt personalizzati ora combinano con template preset per flessibilità migliorata.  
* **Cache Intelligente Prompt**: Sistema cache automatico previene rigenerazione di prompt identici, migliorando dramaticamente performance per input ripetuti. La cache persiste riavvii ComfyUI.  
* **🎛️ Bypass Mode**: Nuovo parametro `bypass_mode` permette di mantenere i prompt generati in precedenza senza rigenerarli. Genera una volta, poi abilita bypass mode per preservare i prompt cambiando gli input. Zero risorse in modalità bypass.  
* **Modalità Seed Fisso**: Imposta seed = 1 per ignorare cambiamenti immagine/video e mantenere prompt consistenti indipendentemente dalle variazioni multimediali. Perfetto per output workflow stabili.  
* **Integrazione WAN 2.2**: Prompt specializzati per generazione WAN 2.2 I2V (immagine-a-video) e T2V (testo-a-video) con specifiche cinematografiche professionali e struttura timeline cinematografica. Preset I2V prioritizzato per accesso workflow più rapido.  
* **Cinematografia Professionale**: Tutti i preset WAN 2.2 includono specifiche tecniche complete - sorgenti luminose, tipi ripresa, specifiche lenti, movimenti camera e requisiti tonalità colore per generazione video professionale.  
* **Storyboard Esteso**: Nuovo preset per generazione storyboard-to-storyboard senza soluzione di continuità con compatibilità formato WAN 2.2, focus continuità e dettagli cinematografia professionale.  
* **Branding Famiglia WAN**: Nomi coerenti su tutti i preset WAN 2.2 per migliore organizzazione e chiarezza workflow.  
* **Modelli Abliterated Free**: Modelli predefiniti includono opzioni non censurate senza token (Qwen3-4B-abliterated-TIES, Qwen3-8B-abliterated-TIES) per accessibilità immediata.  
* **Supporto Multi-Modello**: Cambia facilmente tra vari modelli ufficiali Qwen-VL con ordinamento intelligente 4B-primo per efficienza VRAM.  
* **Download Automatico Modelli**: I modelli sono scaricati automaticamente al primo utilizzo.  
* **Quantizzazione Intelligente**: Bilancia VRAM e performance con opzioni 4-bit, 8-bit e FP16. Quantizzazione 8-bit abilitata predefinita per accessibilità ottimale.  
* **Attention Ottimizzato**: Pipeline attention pulita con supporto Flash Attention 2 e fallback stabile SDPA. Nessun patching complesso che potrebbe interferire con output modello.  
* **Hardware-Aware**: Rileva automaticamente capacità GPU e previene errori con modelli incompatibili (es. FP8).  
* **Generazione Riproducibile**: Usa parametro seed per output consistenti, con Modalità Seed Fisso per stabilità ultima.  
* **Gestione Memoria**: Opzione "Keep Model Loaded" per mantenere modello in VRAM per elaborazione più rapida.  
* **Supporto Immagine & Video**: Accetta sia immagini singole che sequenze frame video come input.  
* **Gestione Errori Robusta**: Fornisce messaggi errore chiari per problemi hardware o memoria.  
* **Output Console Pulito**: Log console minimi e informativi durante operazione.

## **🚀 Installazione**

1. Clona questo repository nella tua directory ComfyUI/custom_nodes:  
   ```
   cd ComfyUI/custom_nodes  
   git clone https://github.com/huchukato/ComfyUI-QwenVL-Mod.git
   ```
2. Installa le dipendenze richieste:  
   ```
   cd ComfyUI/custom_nodes/ComfyUI-QwenVL-Mod  
   pip install -r requirements.txt
   ```

3. Riavvia ComfyUI.

### **Installazione Opzionale: SageAttention**

Per 2-5x boost performance con GPU compatibili:

```bash
# Installa SageAttention 2.2.0 (raccomandato)
pip install sageattention==2.2.0 --no-build-isolation

# O compila da sorgente
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
python setup.py install
```

**Requisiti per SageAttention:**
- NVIDIA GPU con capability >= 8.0 (serie RTX 30/40/50)
- CUDA >= 12.0
- PyTorch >= 2.3.0

Vedi [sezione SageAttention](#-boost-performance-sageattention) per dettagli.

## **🧭 Panoramica Nodi**

### **Nodi Transformers (HF)**
- **QwenVL**: Inferenza vision-linguaggio rapida (immagine/video + preset/prompt personalizzati).  
- **QwenVL (Avanzato)**: Controllo completo su parametri sampling, device e performance.  
- **QwenVL Prompt Enhancer**: Miglioramento prompt solo testo (supporta sia modelli testo Qwen3 che modelli QwenVL in modalità testo).  

### **Nodi GGUF (llama.cpp)**
- **QwenVL (GGUF)**: Inferenza vision-linguaggio GGUF.  
- **QwenVL (GGUF Avanzato)**: Controlli GGUF estesi (contesto, layer GPU, ecc.).  
- **QwenVL Prompt Enhancer (GGUF)**: Miglioramento prompt solo testo GGUF.  

## **🧩 Nodi GGUF (backend llama.cpp)**

Questo repository include nodi **GGUF** powered by `llama-cpp-python` (separati dai nodi basati Transformers).

- **Nodi**: `QwenVL (GGUF)`, `QwenVL (GGUF Avanzato)`, `QwenVL Prompt Enhancer (GGUF)`
- **Cartella modelli** (predefinito): `ComfyUI/models/llm/GGUF/` (configurabile via `gguf_models.json`)
- **Requisito vision**: installa wheel `llama-cpp-python` con capacità vision che fornisce `Qwen3VLChatHandler` / `Qwen25VLChatHandler`  
  Vedi [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md)

## **🗂️ File Configurazione**

- **Modelli HF**: `hf_models.json`  
  - `hf_vl_models`: modelli vision-linguaggio (usati da nodi QwenVL).  
  - `hf_text_models`: modelli solo testo (usati da Prompt Enhancer).  
- **Modelli GGUF**: `gguf_models.json`  
- **Prompt di sistema**: `AILab_System_Prompts.json` (include sia prompt VL che stili prompt-enhancer).  

## **📥 Download Modelli**

I modelli saranno scaricati automaticamente al primo utilizzo. Se preferisci scaricarli manualmente, mettili nella directory ComfyUI/models/LLM/Qwen-VL/.

### **Modelli Vision HF (Qwen-VL)**
| Modello | Link |
| :---- | :---- |
| Qwen3-VL-2B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| Qwen3-VL-2B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) |
| Qwen3-VL-2B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8) |
| Qwen3-VL-2B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8) |
| Qwen3-VL-4B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen3-VL-4B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) |
| Qwen3-VL-4B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8) |
| Qwen3-VL-4B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-FP8) |
| Qwen3-VL-8B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-8B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) |
| Qwen3-VL-8B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) |
| Qwen3-VL-8B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-FP8) |
| Qwen3-VL-32B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL-32B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking) |
| Qwen3-VL-32B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) |
| Qwen3-VL-32B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8) |
| Qwen2.5-VL-3B-Instruct | [Download](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL-7B-Instruct | [Download](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

### **Modelli Testo HF (Qwen3)**
| Modello | Link |
| :---- | :---- |
| Qwen3-0.6B | [Download](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-4B-Instruct-2507 | [Download](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| qwen3-4b-Z-Image-Engineer | [Download](https://huggingface.co/BennyDaBall/qwen3-4b-Z-Image-Engineer) |

### **Modelli GGUF (Download Manuale)**
| Gruppo | Modello | Repo | Alt Repo | File Modelli | MMProj |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Qwen testo (GGUF) | Qwen3-4B-GGUF | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |  | Qwen3-4B-Q4_K_M.gguf, Qwen3-4B-Q5_0.gguf, Qwen3-4B-Q5_K_M.gguf, Qwen3-4B-Q6_K.gguf, Qwen3-4B-Q8_0.gguf |  |
| Qwen-VL (GGUF) | Qwen3-VL-4B-Instruct-GGUF | [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) |  | Qwen3VL-4B-Instruct-F16.gguf, Qwen3VL-4B-Instruct-Q4_K_M.gguf, Qwen3VL-4B-Instruct-Q8_0.gguf | mmproj-Qwen3VL-4B-Instruct-F16.gguf |
| Qwen-VL (GGUF) | Qwen3-VL-8B-Instruct-GGUF | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |  | Qwen3VL-8B-Instruct-F16.gguf, Qwen3VL-8B-Instruct-Q4_K_M.gguf, Qwen3VL-8B-Instruct-Q8_0.gguf | mmproj-Qwen3VL-8B-Instruct-F16.gguf |
| Qwen-VL (GGUF) | Qwen3-VL-4B-Thinking-GGUF | [Qwen/Qwen3-VL-4B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-GGUF) |  | Qwen3VL-4B-Thinking-F16.gguf, Qwen3VL-4B-Thinking-Q4_K_M.gguf, Qwen3VL-4B-Thinking-Q8_0.gguf | mmproj-Qwen3VL-4B-Thinking-F16.gguf |
| Qwen-VL (GGUF) | Qwen3-VL-8B-Thinking-GGUF | [Qwen/Qwen3-VL-8B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-GGUF) |  | Qwen3VL-8B-Thinking-F16.gguf, Qwen3VL-8B-Thinking-Q4_K_M.gguf, Qwen3VL-8B-Thinking-Q8_0.gguf | mmproj-Qwen3VL-8B-Thinking-F16.gguf |

## **📖 Utilizzo**

### **Uso Base**

1. Aggiungi il nodo **"QwenVL"** dalla categoria 🧪AILab/QwenVL.  
2. Seleziona il **model\_name** che desideri utilizzare.  
3. Connetti un'immagine o video (sequenza frame) al nodo.  
4. Scrivi il tuo prompt usando campo preset o personalizzato.  
5. Esegui il workflow.

### **Uso Avanzato**

Per più controllo, usa il nodo **"QwenVL (Avanzato)"**. Questo ti dà accesso a parametri generazione dettagliati come temperatura, top\_p, beam search e selezione device.

## **⚙️ Parametri**

| Parametro | Descrizione | Predefinito | Range | Nodi |
| :---- | :---- | :---- | :---- | :---- |
| **model\_name** | Il modello Qwen-VL da usare. | Qwen3-VL-4B-Instruct | \- | Standard & Avanzato |
| **quantization** | Quantizzazione al volo. Ignorata per modelli pre-quantizzati (es. FP8). | 8-bit (Bilanciato) | 4-bit, 8-bit, None | Standard & Avanzato |
| **preset\_prompt** | Selezione di prompt predefiniti per compiti comuni. | "Describe this..." | Testo qualsiasi | Standard & Avanzato |
| **custom\_prompt** | Sostituisce preset prompt se fornito. |  | Testo qualsiasi | Standard & Avanzato |
| **max\_tokens** | Numero massimo nuovi token da generare. | 1024 | 64-2048 | Standard & Avanzato |
| **keep\_model\_loaded** | Mantiene modello in VRAM per esecuzioni successive più veloci. | True | True/False | Standard & Avanzato |
| **seed** | Seed per risultati riproducibili. | 1 | 1 \- 2^64-1 | Standard & Avanzato |
| **temperature** | Controlla casualità. Valori più alti = più creativi. (Usato quando num\_beams è 1). | 0.6 | 0.1-1.0 | Solo Avanzato |
| **top\_p** | Soglia sampling nucleo. (Usato quando num\_beams è 1). | 0.9 | 0.0-1.0 | Solo Avanzato |
| **num\_beams** | Numero di beam per beam search. > 1 disabilita sampling temperature/top\_p. | 1 | 1-10 | Solo Avanzato |
| **repetition\_penalty** | Scoraggia token ripetuti. | 1.2 | 0.0-2.0 | Solo Avanzato |
| **frame\_count** | Numero di frame da campionare da input video. | 16 | 1-64 | Solo Avanzato |
| **device** | Sovrascrive selezione device automatica. | auto | auto, cuda, cpu | Solo Avanzato |
| **attention_mode** | Backend attention per ottimizzazione performance. | auto | auto, flash_attention_2, sdpa | Standard & Avanzato |

### **💡 Opzioni Quantizzazione**

| Modalità | Precisione | Uso Memoria | Velocità | Qualità | Raccomandato Per |
| :---- | :---- | :---- | :---- | :---- | :---- |
| None (FP16) | Float 16-bit | Alto | Velocissimo | Migliore | GPU VRAM alta (16GB+) |
| 8-bit (Bilanciato) | Intero 8-bit | Medio | Veloce | Ottimo | Performance bilanciata (8GB+) |
| 4-bit (VRAM-friendly) | Intero 4-bit | Basso | Più lento* | Buono | GPU VRAM bassa (<8GB) |

\* **Nota Velocità 4-bit**: La quantizzazione 4-bit riduce significativamente uso VRAM ma può risultare in performance più lente su alcuni sistemi a causa overhead computazionale dequantizzazione real-time.

### **⚡ Opzioni Modalità Attention**

| Modalità | Descrizione | Velocità | Memoria | Requisiti |
| :---- | :---- | :---- | :---- | :---- |
| **auto** | Seleziona automaticamente FlashAttention se disponibile, fallback SDPA | Veloce | Medio | pacchetto flash-attn |
| **flash_attention_2** | Usa FlashAttention v2 per performance ottimali | Velocissimo | Basso | flash-attn + GPU CUDA |
| **sdpa** | PyTorch nativo Scaled Dot Product Attention | Medio | Medio | PyTorch 2.0+ |
| **sageattention** | Attention quantizzato 8-bit con 2-5x velocità | Molto Veloce | Più basso | sageattention + GPU CUDA |

**Requisiti SageAttention:**
- NVIDIA GPU con capability >= 8.0 (serie RTX 30/40/50)
- CUDA >= 12.0
- PyTorch >= 2.3.0
- pacchetto sageattention installato

### **🤔 Consigli Impostazioni**

| Impostazione | Raccomandazione |
| :---- | :---- |
| **Scelta Modello** | Per molti utenti, Qwen3-VL-4B-Instruct è ottimo punto di partenza. Se hai GPU serie 40, prova versione -FP8 per performance migliori. |
| **Modalità Memoria** | Mantieni keep\_model\_loaded abilitato (True) per performance migliori se prevedi eseguire nodo più volte. Disabilitalo solo se esaurisci VRAM per altri nodi. |
| **Quantizzazione** | Inizia con predefinito 8-bit. Se hai VRAM abbondante (>16GB), passa a None (FP16) per velocità e qualità migliori. Se sei basso in VRAM, usa 4-bit. |
| **Performance** | Prima volta caricamento modello con quantizzazione specifica, può essere lento. Esecuzioni successive (con keep\_model\_loaded abilitato) saranno molto più veloci. |
| **Modalità Attention** | Usa "sageattention" per 2-5x velocità se hai GPU compatibile. Altrimenti usa "auto" per selezione automatica. |

## **🧠 Informazioni Modello**

Questo nodo utilizza serie modelli Qwen-VL, sviluppati dal Qwen Team di Alibaba Cloud. Questi sono potenti modelli vision-linguaggio open-source (LVLM) progettati per capire e processare informazioni sia visive che testuali, rendendoli ideali per compiti come descrizione dettagliata immagini e video.

## **⚡ Boost Performance SageAttention**

Questa integrazione include supporto per **SageAttention**, un'implementazione attention quantizzata 8-bit all'avanguardia che fornisce significativi miglioramenti performance:

### **🚀 Guadagni Performance**

| Modello | FlashAttention | SageAttention | Velocità |
|-------|----------------|--------------|---------|
| Qwen2.5-VL-3B | 100% | 250-400% | 2.5-4x |
| Qwen3-VL-4B | 100% | 200-500% | 2-5x |
| Qwen3-VL-8B | 100% | 300-500% | 3-5x |

### **🎯 Come Usare**

1. **Installa SageAttention** (vedi [Installazione](#-installazione-opzionale-sageattention))
2. **Seleziona "sageattention"** nel parametro `attention_mode`
3. **Esegui il tuo workflow** - il sistema applica automaticamente l'ottimizzazione

### **🔧 Dettagli Tecnici**

- **Implementazione**: Usa monkey patching per sostituire `scaled_dot_product_attention` di PyTorch
- **Compatibilità**: Funziona con tutte modalità quantizzazione (4-bit, 8-bit, FP16)
- **Integrazione**: Si integra senza soluzione di continuità con workflow esistenti
- **Fallback**: Ripiega automaticamente su SDPA se SageAttention non è disponibile

### **📋 Checklist Requisiti**

- [ ] NVIDIA GPU con capability >= 8.0 (serie RTX 30/40/50)
- [ ] CUDA >= 12.0
- [ ] PyTorch >= 2.3.0
- [ ] pacchetto sageattention installato
- [ ] VRAM sufficiente per modello scelto

### **🐛 Risoluzione Problemi**

**SageAttention non funziona?**
```bash
# Controlla installazione
python -c "import sageattention; print('SageAttention disponibile')"

# Controlla capability GPU
python -c "import torch; print(f'GPU capability: {torch.cuda.get_device_capability()}')"
```

**Problemi Comuni:**
- **"SageAttention non disponibile"**: Installa il pacchetto e controlla compatibilità GPU
- **"CUDA non disponibile"**: Assicurati di avere installazione PyTorch compatibile CUDA
- **"GPU capability insufficiente"**: SageAttention richiede serie RTX 30 o più recente

### **📚 Riferimenti**

- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [SageAttention Paper](https://arxiv.org/abs/2410.02367)
- [Benchmark Performance](https://github.com/thu-ml/SageAttention#performance)

## **🎬 Integrazione WAN 2.2**

Questa versione migliorata include prompt specializzati per **WAN 2.2** generazione video, supportando sia workflow I2V (immagine-a-video) che T2V (testo-a-video).

### **🎯 Prompt WAN 2.2 Disponibili**

| Tipo Prompt | Caso Uso | Input | Output | Posizione |
|:---|:---|:---|:---|:---|
| **🍿 Wan 2.2 I2V** | Immagine-a-Video | Immagine + Testo | Timeline cinematografica 5-secondi | Nodi QwenVL |
| **🍿 Wan 2.2 T2V** | Testo-a-Video | Testo solo | Timeline cinematografica 5-secondi | Nodi Prompt Enhancer |

### **⚡ Funzionalità**

- **Struttura Timeline Cinematografica**: Video 5-secondi con descrizioni secondo per secondo
- **Supporto Multilingua**: Input Italiano/Inglese → Output ottimizzato Inglese
- **Descrizione Scena Professionale**: Direzione stile film inclusi illuminazione, camera, composizione
- **Gestione NSFW**: Filtro contenuti e descrizione appropriati
- **Ottimizzazione WAN 2.2**: Formattato specificamente per migliori risultati generazione video

### **📝 Esempio Formato Output**

```
(At 0 seconds: A young woman stands facing a rack of clothes...)
(At 1 second: The blouse falls to the floor around her feet...)
(At 2 seconds: She reaches out with her right hand...)
(At 3 seconds: She turns her body slightly towards the mirror...)
(At 4 seconds: Lifting the hanger, she holds the dark fabric...)
(At 5 seconds: A subtle, thoughtful expression crosses her face...)
```

### **🔧 Utilizzo**

1. **Per I2V**: Usa preset "🍿 Wan 2.2 I2V" in nodi QwenVL con input immagine
2. **Per T2V**: Usa stile "🍿 Wan 2.2 T2V" in nodi Prompt Enhancer con solo testo
3. **Per Storyboard**: Usa "🍿 Wan Extended Storyboard" per continuità scena senza soluzione di continuità
4. **Per Video Generale**: Usa "🎥 Wan Cinematic Video" per descrizioni scena professionali singole

### **🎨 Best Practices**

- Fornisci input chiaro e descrittivo per migliore interpretazione scena
- Usa direzioni camera e illuminazione specifiche quando possibile
- Includi dettagli umore e atmosfera per risultati cinematografici
- Sfrutta specifiche cinematografia professionale per qualità video ottimale
- Il sistema gestisce automaticamente ottimizzazione timeline per preset WAN 2.2

## **🗺️ Roadmap**

### **✅ Completato (v2.0.7)**

* ✅ Supporto per modelli Qwen3-VL e Qwen2.5-VL.  
* ✅ Supporto backend GGUF per inferenza più rapida.  
* ✅ Nodi Prompt Enhancer per workflow solo testo.  
* ✅ Integrazione Flash Attention 2 per 2-3x boost performance.  
* ✅ Prompt generazione video WAN 2.2 I2V e T2V.  
* ✅ Preset Storyboard Esteso per continuità scena.  
* ✅ Specifiche cinematografia professionale per tutti preset WAN 2.2.  
* ✅ Branding famiglia WAN e nomi coerenti.  
* ✅ Preset Storyboard Esteso per generazione continuità senza soluzione di continuità.  
* ✅ Modelli abliterated free senza requisiti token.  
* ✅ Logica prompt personalizzata migliorata su tutti i nodi.  
* ✅ Supporto completo generazione contenuti NSFW.  
* ✅ Ordinamento modelli ottimizzato e predefiniti quantizzazione.  
* ✅ Pipeline attention pulita con stabilità SDPA.  
* ✅ Rimossa complessità per migliore affidabilità output modello.  
* ✅ Sistema cache intelligente prompt per ottimizzazione performance.  
* ✅ Modalità Seed Fisso per output stabili indipendentemente da variazioni multimediali.  
* ✅ Cache persistente riavvii ComfyUI.  
* ✅ Aggiornamenti manutenzione codice per compatibilità futura.

## **🙏 Crediti**

* **Qwen Team**: [Alibaba Cloud](https://github.com/QwenLM) - Per sviluppo e open-source potenti modelli Qwen-VL.  
* **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI) - Per incredibile e estensibile piattaforma ComfyUI.  
* **llama-cpp-python**: [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) - Backend GGUF con supporto vision usato da nodi GGUF.  
* **Integrazione ComfyUI**: [1038lab](https://github.com/1038lab) - Sviluppatore di questo custom node.

## **👥 Autore**

- **huchukato**
  - 🐙 [GitHub](https://github.com/huchukato)
  - 🐦 [X (Twitter)](https://twitter.com/huchukato)
  - 🎨 [Civitai](https://civitai.com/user/huchukato) - Dai un'occhiata ai miei modelli arte AI!

## **📜 Licenza**

Il codice di questo repository è rilasciato sotto [Licenza GPL-3.0](LICENSE).
