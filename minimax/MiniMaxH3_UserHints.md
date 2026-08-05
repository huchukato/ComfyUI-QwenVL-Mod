# 🎬 MiniMax H3 — User Hints

> 💡 You can write your prompt in **any language**. Qwen3-VL analyzes it, translates it, and converts it into the official MiniMax H3 prompt format.

---

## 🗂️ Choose the right workflow

| Workflow | Mode | Required inputs | Preset icon |
|---|---|---|---|
| `MiniMaxH3-T2VA-Qwen3VL.json` | 📝 **T2VA** | text only | 🎬 |
| `MiniMaxH3-I2VA-Qwen3VL.json` | 🖼️ **I2VA** | text + first-frame image (`image`) | 🎬 |
| `MiniMaxH3-FL2VA-Qwen3VL.json` | 🔄 **FL2VA** | text + first-frame (`image`) + last-frame (`image2`) | 🔄 |
| `MiniMaxH3-R2VA-Qwen3VL.json` | 🎞️ **R2VA** | text + reference images (`image` + `image2`) | 🎞️ |

> 🧩 **L2VA** (last frame only) is handled by the I2VA preset when you connect only the last frame.

---

## 🎯 Choose the right preset

| Icon | Preset | Mode | Format |
|---|---|---|---|
| 🎬 | MiniMax H3 NSFW (5s/10s/15s) | T2VA / I2VA | 3 fields: `integrated_multimodal_description` + `overall_soundscape` + `non_diegetic_music` |
| 🔄 | MiniMax H3 NSFW FL2VA (5s/10s/15s) | FL2VA | 3 fields, transition-focused (describes the path between frames, not the scene) |
| 🎞️ | MiniMax H3 NSFW R2VA (5s/10s/15s) | R2VA | 6 fields: `subject_definitions` + `summary` + `retention_analysis` + `detailed_description` + `overall_soundscape` + `non_diegetic_music` |

> ⚠️ **Use the correct preset for each mode!** R2VA and FL2VA have dedicated presets that follow the official MiniMax H3 prompt writing guides.

---

## 🖼️ How to connect images

### T2VA (text only)
- No images needed
- Uses `AILab_QwenVL_PromptEnhancer` (text-only node)

### I2VA (first-frame)
| Input | What to connect |
|---|---|
| `image` | First-frame image |
| `image2` | (empty) |

### FL2VA (first + last frame)
| Input | What to connect | `frame_count` |
|---|---|---|
| `image` | First-frame image | — |
| `image2` | Last-frame image | 1 |

> 💡 Qwen3-VL sees **both** images and describes the transition between them.

### R2VA (reference)
| Input | What to connect | `frame_count` |
|---|---|---|
| `image` | Primary reference (character, style, scene) | — |
| `image2` | Additional references (batch, up to 9) | 1–9 |

> 💡 Qwen3-VL sees **all** reference images. Reference them by order in your prompt: `<Picture 1>`, `<Picture 2>`, etc.

---

## ✍️ What to write in the prompt

Describe the scene naturally. Be clear about the concepts below — the model handles the rest.

### 🎨 1. Visual style *(required — put it first)*

- `photorealistic` 📷 · `cinematic` 🎥 · `live-action` 🎬
- `anime` 🌸 · `cartoon` 🖍️ · `3D CG` 🧊 · `claymation` 🏺
- `vintage film` 📼 · `watercolor` 🎨 · `fantasy` ✨ · `artistic portrait` 🖼️

> Example: *"A photorealistic cinematic scene in a bedroom with warm light..."*

### 👥 2. Subjects

- Number, gender, apparent age, physical appearance, hair, makeup, clothing (or nudity)
- Initial position, gaze, expression

### 🏃 3. Action / motion

- What happens and in what order
- Speed: slow 🐢 · rhythmic 🎵 · accelerating 🚀 · pause ⏸️
- Interaction between characters: contact, gestures, movements

> 🔄 **FL2VA**: Describe the **transition** between frames — how subjects move, poses change, composition evolves. Do NOT re-describe the scene (the images already fix it).

> 🎞️ **R2VA**: Reference your inputs by tag: `<Picture 1>`, `<Picture 2>`, `<Video 1>`, `<Audio 1>`. State what each reference controls (identity, style, motion, voice).

### 📷 4. Camera

- **Shot type**: `close-up` 🔍 · `medium shot` 🎥 · `wide shot` 🌅 · `POV` 👁️
- **Motion**: `static shot` · `push in` / `pull out` · `pan left/right` · `tilt up/down` · `tracking shot` · `arc shot` · `zoom in/out`
- **Speed & amplitude**: slow/fast, small amplitude/large amplitude

> Example: *"The camera starts with a static medium-wide shot, then slowly pushes in toward the face."*

### 🌆 5. Environment & lighting

- 📍 Location: bedroom, bathroom, couch, outdoor, night, neon, natural
- 💡 Light: warm, cold, pink/red neon, windows, shadows
- 🎨 Dominant color and atmosphere

---

## 🎧 6. Audio *(very important)*

MiniMax H3 generates **native audio**. Explicitly state what you want to hear.

### 🔊 Diegetic sounds *(present in the scene)*

Always described in the prompt. Be concrete:

| Category | Examples |
|---|---|
| 🫁 Breathing | soft, heavy, gasping, rhythmic |
| 😏 Vocal | moans, sighs, gasps, whimpers, whispers |
| 💓 Body | heartbeat, skin-to-skin contact, body fluids |
| 🛏️ Movement | fabric, bedding, sheets rustling |
| 💋 Contact | kisses, licks, slaps, impacts |
| 🌧️ Ambient | water, rain, distant traffic, room tone, wind |

> Example: *"Heavy breathing, soft moans, and the sound of sheets rustling."*

### 🎵 Non-diegetic music *(background score)*

By default the preset outputs **`N/A`** — no background music is generated. 🚫🎵

To add music, you must **explicitly request it** and describe:

- 🎼 **Genre**: R&B, ambient, synthwave, jazz, deep house, orchestral, lo-fi...
- 🎸 **Instrumentation**: piano, strings, bass, synths, guitar, drums
- ⏱️ **Tempo**: slow, moderate, fast
- 📈 **Rhythm & dynamics**: when it enters, builds, fades

> Examples:
> - *"Add a slow R&B background track with deep bass and atmospheric synths."*
> - *"Soft piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume before fading out."*
> - *"No music, only realistic sounds from the scene."* ← this is the **default** behavior

> 💡 **Tip**: For NSFW scenes, diegetic sounds alone usually feel more realistic. Add music only when it clearly supports the mood.

---

## ⏱️ 7. Duration

Pick the matching QwenVL-Mod preset:

| Preset | Duration |
|---|---|
| 🎬 / 🔄 / 🎞️ MiniMax H3 NSFW (5s) | 5 seconds |
| 🎬 / 🔄 / 🎞️ MiniMax H3 NSFW (10s) | 10 seconds |
| 🎬 / 🔄 / 🎞️ MiniMax H3 NSFW (15s) | 15 seconds |

> MiniMax H3 supports clips from **4 to 15 seconds**.

---

## 📐 Recommended resolution

MiniMax H3 is trained with the **short edge at 768 px** and the long edge **capped at 1344 px**, in multiples of 32.

| Resolution | Orientation |
|---|---|
| `768x1344` | 📱 portrait |
| `896x1152` | 📱 portrait |
| `960x1280` | 📱 portrait |
| `1024x1024` | ⬛ square |
| `1344x768` | 🖥️ landscape |
| `1152x896` | 🖥️ landscape |
| `1280x960` | 🖥️ landscape |

> ⚠️ **Match the aspect ratio to your input image!** If your image is portrait (e.g. 832×1216), select a portrait resolution (e.g. 768×1344). Forcing 16:9 on a portrait image will squash it.
>
> ⚠️ **Avoid generating directly at 1080p.** Generate at native resolution, then use the bundled TensorRT upscale/interpolation nodes (I2VA + FL2VA workflows).

---

## 📝 Example prompts

### T2VA / I2VA
> *"A photorealistic cinematic scene in a bedroom with warm lamplight. A young dark-haired woman lies on the bed wearing only white sheets. A man approaches slowly; the camera does a gentle push in from wide shot to close-up. He kisses her neck, she closes her eyes and sighs. Audio: heavy breathing, whispers, sheets rustling. No background music. Intimate, realistic style, warm light."*

### FL2VA
> *"The woman is lying on the bed in the first frame and sitting up in the last frame. Describe the transition: she slowly rises, the sheets slide off, the camera tilts up with her movement. Audio: sheets rustling, soft gasp as she sits up. No music."*

### R2VA
> *"<Picture 1> is the character reference — a young woman with red hair. <Picture 2> is the environment — a luxury bathroom. Generate a scene where the woman from <Picture 1> is in the environment from <Picture 2>, relaxing in the tub. Audio: water splashing, soft sighs. No music."*

---

## 🚫 What NOT to do

- ❌ Do not request minors, non-consent, or illegal acts — the preset refuses automatically
- ❌ Do not add lighting or effects inconsistent with the described environment
- ❌ Do not request durations longer than 15 seconds
- ❌ Do not request 1080p native generation — use the recommended resolutions and upscale after
- ❌ Do not use a 🎬 base preset for R2VA or FL2VA — use the dedicated 🎞️ / 🔄 presets
- ❌ Do not force 16:9 resolution on a portrait image — match the aspect ratio
