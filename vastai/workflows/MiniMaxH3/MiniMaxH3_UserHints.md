# MiniMax H3 - Hint per l'utente

Puoi scrivere il prompt in italiano (o in qualsiasi lingua). Qwen3-VL lo analizza, lo traduce e lo converte nel formato ufficiale MiniMax H3.

## Scegli il workflow corretto

| Workflow | Modalità | Input richiesti |
|---|---|---|
| `MiniMaxH3-T2VA-Qwen3VL.json` | **T2VA** | solo testo |
| `MiniMaxH3-I2VA-Qwen3VL.json` | **I2VA** | testo + immagine primo frame |
| `MiniMaxH3-R2VA-Qwen3VL.json` | **R2VA / Reference** | testo + immagine/video/audio di riferimento per stile, personaggio, movimento o camera |

> **FL2VA** (primo frame + ultimo frame) e **L2VA** (solo ultimo frame) sono gestiti automaticamente dal preset se carichi più di un'immagine o indichi chiaramente l'ultimo frame.

## Cosa scrivere nel prompt

Descrivi la scena in modo naturale. Non serve usare inglese: basta che siano chiari i concetti.

### 1. Stile visivo (obbligatorio, mettilo all'inizio)

- `photorealistic`, `cinematic`, `live-action`
- `anime`, `cartoon`, `3D CG`, `claymation`
- `vintage film`, `watercolor`, `fantasy`, `artistic portrait`

Esempio: *"Una scena photorealistic cinematografica in una camera da letto con luce calda..."*

### 2. Soggetti

- Numero, genere, età apparente, aspetto fisico, capelli, trucco, abiti (o nudità)
- Posizione iniziale, sguardo, espressione

### 3. Azione / movimento

- Cosa succede e in che ordine
- Velocità: lento, ritmico, accelerazione, pausa
- Interazione tra personaggi: contatto, gesti, spostamenti

### 4. Camera

- Tipo di inquadratura: `close-up`, `medium shot`, `wide shot`, `POV`
- Movimento: `static shot`, `push in`, `pull out`, `pan left/right`, `tilt up/down`, `tracking shot`, `arc shot`, `zoom in/out`
- Velocità e ampiezza: lento/veloce, piccola ampiezza/grande ampiezza

Esempio: *"La camera parte da un medium-wide shot statico e poi fa un push in lento verso il viso."*

### 5. Ambiente e luce

- Luogo: camera, bagno, divano, esterno, notturno, neon, naturale
- Luce: luce calda, fredda, neon rosa/rosso, finestre, ombre
- Colore dominante e atmosfera

### 6. Audio (molto importante)

MiniMax H3 genera audio nativo. Specifica esplicitamente cosa vuoi sentire:

- **Suoni diegetici** (presenti nella scena): respiri, gemiti, sospiri, battiti del cuore, contatto pelle-pelle, tessuti, letto, liquidi, voci sussurrate, ambiente (acqua, pioggia, traffico lontano).
- **Musica di sottofondo**: se la vuoi, indica genere, strumenti, ritmo e intensità. **Se non la menzioni, il preset imposta `N/A` e non genera musica generica.**

Esempi:
- *"Si sentono respiri affannosi, gemiti sommessi e il rumore delle lenzuola."*
- *"Aggiungi una colonna sonora R&B lenta con basso profondo e synth atmosferici."*
- *"Nessuna musica, solo i suoni realistici della scena."*

### 7. Durata

Scegli il preset QwenVL-Mod corrispondente alla durata desiderata:

- **MiniMax H3 NSFW (5s)**
- **MiniMax H3 NSFW (10s)**
- **MiniMax H3 NSFW (15s)**

MiniMax H3 supporta clip da **4 a 15 secondi**.

## Risoluzione consigliata

MiniMax H3 è addestrato con il **lato corto a 768 px** e il lato lungo **massimo 1344 px**, in multipli di 32.

Esempi validi:

- `768x1344` (verticale)
- `896x1152`
- `960x1280`
- `1024x1024`

**Evita di generare direttamente a 1080p.** Genera a risoluzione nativa e poi usa i nodi TensorRT di upscale/interpolazione inclusi nel template.

## Esempio di prompt italiano

> *"Scena photorealistic cinematografica in una camera da letto con luce calda di lampada da comodino. Una giovane donna dai capelli scuri è sdraiata sul letto, indossa solo lenzuola bianche. Un uomo si avvicina lentamente, la camera fa un push in morbido dal wide shot al close-up. Lui la bacia sul collo, lei chiude gli occhi e sospira. Audio: respiri affannosi, sussurri, rumore delle lenzuola. Nessuna musica di sottofondo. Stile intimo, realistico, luce calda."*

## Cosa NON mettere

- Non richiedere personaggi minorenni o scene non consensuali/illegali: il preset rifiuta automaticamente.
- Non aggiungere luci o effetti che non siano coerenti con l'ambiente descritto.
- Non chiedere durate superiori a 15 secondi.
