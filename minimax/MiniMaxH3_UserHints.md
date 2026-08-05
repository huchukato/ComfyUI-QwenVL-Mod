# 🎬 MiniMax H3 — User Hints

> 💡 You can write your prompt in **any language**. Qwen3-VL analyzes it, translates it, and converts it into the official MiniMax H3 prompt format.

---

## 🗂️ Choose the right workflow

| Workflow | Mode | Required inputs |
|---|---|---|
| `MiniMaxH3-T2VA-Qwen3VL.json` | 📝 **T2VA** | text only |
| `MiniMaxH3-I2VA-Qwen3VL.json` | 🖼️ **I2VA** | text + first-frame image |
| `MiniMaxH3-R2VA-Qwen3VL.json` | 🎞️ **R2VA / Reference** | text + reference image/video/audio for style, character, motion, or camera |

> 🧩 **FL2VA** (first frame + last frame) and **L2VA** (last frame only) are handled automatically by the preset when you load more than one image or clearly indicate the last frame.

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
| 🎬 MiniMax H3 NSFW (5s) | 5 seconds |
| 🎬 MiniMax H3 NSFW (10s) | 10 seconds |
| 🎬 MiniMax H3 NSFW (15s) | 15 seconds |

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

> ⚠️ **Avoid generating directly at 1080p.** Generate at native resolution, then use the bundled TensorRT upscale/interpolation nodes.

---

## 📝 Example prompt

> *"A photorealistic cinematic scene in a bedroom with warm lamplight. A young dark-haired woman lies on the bed wearing only white sheets. A man approaches slowly; the camera does a gentle push in from wide shot to close-up. He kisses her neck, she closes her eyes and sighs. Audio: heavy breathing, whispers, sheets rustling. No background music. Intimate, realistic style, warm light."*

---

## 🚫 What NOT to do

- ❌ Do not request minors, non-consent, or illegal acts — the preset refuses automatically
- ❌ Do not add lighting or effects inconsistent with the described environment
- ❌ Do not request durations longer than 15 seconds
- ❌ Do not request 1080p native generation — use the recommended resolutions and upscale after
