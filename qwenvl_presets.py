"""Shared preset catalog for QwenVL nodes.

Presets live in presets/*.json, one file per family (minimax, wan, ltx,
image, text). Each file may expose two sections:

- "vl":   presets consumed by the VL nodes (preset_prompt widget)
- "text": styles consumed by the Prompt Enhancer nodes (enhancement_style widget)

Entries are either {"text": "..."} or {"durations": {"5s": "...", ...},
"default": "5s"} — duration variants stay grouped under a single dropdown
entry and are selected by the node's "duration" widget.

Internally every entry is flattened to lookup keys of the form
"Name" or "Name (Ns)" so legacy name-based matching keeps working.

A legacy AILab_System_Prompts.json placed next to the node (user
customizations) is overlaid last so its "qwenvl"/"qwen_text" content still
wins over the bundled presets.
"""

import json
import re
import urllib.request
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
PRESETS_DIR = NODE_DIR / "presets"
LEGACY_PATH = NODE_DIR / "AILab_System_Prompts.json"

# Garage (huchukato/ComfyUI-Garage) is the source of truth for preset text —
# synced into presets_remote/ at import so edits ship on ComfyUI restart
# without waiting for a node release or image rebuild. Bundled presets stay
# as offline fallback; remote files win per-key.
_GARAGE_RAW_URL = "https://raw.githubusercontent.com/huchukato/ComfyUI-Garage/master/presets"
_GARAGE_TREE_URL = "https://api.github.com/repos/huchukato/ComfyUI-Garage/git/trees/master?recursive=1"
_REMOTE_DIR = NODE_DIR / "presets_remote"
_MANIFEST = _REMOTE_DIR / ".garage-manifest"

DURATION_OPTIONS = ["5s", "10s", "15s", "20s"]
DEFAULT_DURATION = "5s"

# Legacy dropdown names -> new base preset. Duration suffixes (Ns) are parsed
# separately, so keys omit them.
VL_ALIASES = {
    "MiniMax H3 NSFW": "MiniMax › NSFW I2VA",
    "MiniMax › NSFW": "MiniMax › NSFW I2VA",
    "MiniMax H3 NSFW R2VA": "MiniMax › NSFW R2VA",
    "MiniMax H3 NSFW FL2VA": "MiniMax › NSFW FL2VA",
    "MiniMax › SFW": "MiniMax › SFW I2VA",
    "MiniMax H3 SFW": "MiniMax › SFW I2VA",
    "MiniMax H3 SFW R2VA": "MiniMax › SFW R2VA",
    "MiniMax H3 SFW FL2VA": "MiniMax › SFW FL2VA",
    "Wan 2.2 NSFW I2V": "Wan22 › NSFW I2V",
    "Wan 2.2 NSFW I2V (20s)": "Wan22 › NSFW I2V Long",
    "Wan 2.2 NSFW FL2V": "Wan22 › NSFW FL2V",
    "LTX 2.3 NSFW I2V": "LTX › NSFW I2V",
    "LTX 2.3 NSFW FL2VA": "LTX › NSFW FL2VA",
    "LTX Audio-Input NSFW I2V": "LTX › Audio I2V",
    "Tags": "IMG › Tags",
    "Pony→Natural Language": "IMG › Pony→Natural",
    "Simple Description": "IMG › Simple",
    "Detailed Description": "IMG › Detailed",
    "Ultra Detailed Description": "IMG › Ultra",
    "Cinematic Description": "IMG › Cinematic",
    "Detailed Analysis": "IMG › Analysis",
    "Video Summary": "VID › Summary",
}

TEXT_ALIASES = {
    "MiniMax H3 NSFW": "MiniMax › NSFW T2V",
    "MiniMax › NSFW": "MiniMax › NSFW T2V",
    "MiniMax H3 NSFW R2VA": "MiniMax › NSFW R2VA T2V",
    "MiniMax H3 NSFW FL2VA": "MiniMax › NSFW FL2VA T2V",
    "MiniMax › SFW": "MiniMax › SFW T2V",
    "MiniMax H3 SFW": "MiniMax › SFW T2V",
    "MiniMax H3 SFW R2VA": "MiniMax › SFW R2VA T2V",
    "MiniMax H3 SFW FL2VA": "MiniMax › SFW FL2VA T2V",
    "Wan 2.2 NSFW T2V": "Wan22 › NSFW T2V",
    "Wan 2.2 NSFW T2V (20s)": "Wan22 › NSFW T2V Long",
    "LTX 2.3 NSFW T2V": "LTX › NSFW T2V",
    "Enhance": "Enhance",
    "Refine": "Refine",
    "Creative Rewrite": "Creative Rewrite",
    "Detailed Visual": "Detailed Visual",
    "Artistic Style": "Artistic Style",
    "Technical Specs": "Technical Specs",
    "Pony→Natural Language": "Pony→Natural",
}

_EMOJI_PREFIX = re.compile(r"^[\U0001F000-\U0001FAFF☀-➿⬀-⯿️\u200d\ufe0f]+\s*")
_DURATION_SUFFIX = re.compile(r"\s*\((\d+)s\)\s*$")


def _strip_emoji(name):
    return _EMOJI_PREFIX.sub("", str(name or "")).strip()


def _garage_preset_list():
    """Map of presets/*.json blobs (path -> sha) in the Garage repo.
    None on failure — caller keeps bundled presets."""
    try:
        req = urllib.request.Request(
            _GARAGE_TREE_URL, headers={"User-Agent": "ComfyUI-QwenVL-Mod"})
        tree = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return {
            blob["path"][len("presets/"):]: blob.get("sha", "")
            for blob in tree.get("tree", [])
            if blob.get("type") == "blob"
            and blob["path"].startswith("presets/")
            and blob["path"].endswith(".json")
        }
    except Exception as exc:
        print(f"[QwenVL] Garage preset listing failed, using bundled presets: {exc}")
        return None


def _sync_garage_presets():
    """Blocking mirror of Garage presets/ into presets_remote/ (sha-skipped)."""
    remote = _garage_preset_list()
    if remote is None:
        return
    _REMOTE_DIR.mkdir(exist_ok=True)
    try:
        prev = dict(line.split(" ", 1) for line in
                    _MANIFEST.read_text().splitlines() if " " in line)
    except Exception:
        prev = {}
    for rel, sha in remote.items():
        dest = _REMOTE_DIR / rel
        if prev.get(rel) == sha and dest.is_file():
            continue
        try:
            data = None
            for _ in range(2):
                try:
                    data = urllib.request.urlopen(f"{_GARAGE_RAW_URL}/{rel}", timeout=30).read()
                    break
                except Exception:
                    data = None
            if data is None:
                raise TimeoutError(f"download failed: {rel}")
            if dest.exists() and dest.read_bytes() == data:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"[QwenVL] preset updated from Garage: {rel}")
        except Exception as exc:
            print(f"[QwenVL] preset sync skipped {rel}: {exc}")
    for p in _REMOTE_DIR.rglob("*.json"):
        if p.relative_to(_REMOTE_DIR).as_posix() not in remote:
            p.unlink()
            print(f"[QwenVL] preset removed (gone from Garage): {p.name}")
    try:
        _MANIFEST.write_text("\n".join(f"{rel} {sha}" for rel, sha in sorted(remote.items())))
    except Exception:
        pass


def _load_sections():
    """Merge presets/*.json sections in deterministic file order.
    Bundled files load first; Garage-synced presets_remote/ wins per-key."""
    vl_entries, text_entries = {}, {}
    translation_prompt = ""
    order = ["minimax.json", "wan.json", "ltx.json", "image.json", "text.json"]
    files = []
    for root in (PRESETS_DIR, _REMOTE_DIR):
        files += [root / name for name in order]
        files += sorted(p for p in root.glob("*.json") if p.name not in order)
    for path in files:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, target in (("vl", vl_entries), ("text", text_entries)):
            section = data.get(key)
            if isinstance(section, dict):
                target.update(section)
        if isinstance(data.get("translation_prompt"), str):
            translation_prompt = data["translation_prompt"]
    return vl_entries, text_entries, translation_prompt


def _flatten(entries):
    """Expand {"durations": {...}} entries into flat "Name (Ns)" keys.
    Returns (ordered_base_names, flat_dict, base_durations)."""
    base_names, flat, durations = [], {}, {}
    for name, entry in entries.items():
        base_names.append(name)
        if isinstance(entry, dict) and isinstance(entry.get("durations"), dict):
            durs = entry["durations"]
            durations[name] = (durs.get("default") or next(iter(durs), DEFAULT_DURATION))
            for dur, text in durs.items():
                if isinstance(text, str):
                    flat[f"{name} ({dur})"] = text
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            flat[name] = entry["text"]
        elif isinstance(entry, str):
            flat[name] = entry
    return base_names, flat, durations


def _apply_legacy_overlay(vl_entries, text_entries, translation_prompt):
    """Overlay a user-provided legacy AILab_System_Prompts.json, if present."""
    if not LEGACY_PATH.exists():
        return vl_entries, text_entries, translation_prompt
    try:
        data = json.loads(LEGACY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return vl_entries, text_entries, translation_prompt
    qwenvl = data.get("qwenvl")
    if isinstance(qwenvl, dict):
        for name, text in qwenvl.items():
            if isinstance(text, str):
                vl_entries[name] = {"text": text}
    qwen_text = data.get("qwen_text")
    if isinstance(qwen_text, dict):
        styles = qwen_text.get("styles")
        if isinstance(styles, dict):
            for name, text in styles.items():
                if isinstance(text, str):
                    text_entries[name] = {"text": text}
        if isinstance(qwen_text.get("translation_prompt"), str):
            translation_prompt = qwen_text["translation_prompt"]
    return vl_entries, text_entries, translation_prompt


def _load_all():
    _sync_garage_presets()
    vl_entries, text_entries, translation_prompt = _load_sections()
    vl_entries, text_entries, translation_prompt = _apply_legacy_overlay(
        vl_entries, text_entries, translation_prompt)
    vl_names, vl_flat, vl_durations = _flatten(vl_entries)
    text_names, text_flat, text_durations = _flatten(text_entries)
    return vl_names, vl_flat, vl_durations, text_names, text_flat, text_durations, translation_prompt


(VL_PRESET_NAMES, VL_PROMPTS, VL_DURATIONS,
 TEXT_STYLE_NAMES, TEXT_PROMPTS, TEXT_DURATIONS,
 TRANSLATION_PROMPT) = _load_all()


def resolve_preset(name, aliases):
    """Map any preset name (new, legacy emoji, or duration-suffixed) to
    (canonical_base_name, duration_or_None)."""
    raw = str(name or "").strip()
    if not raw:
        return raw, None
    # Full-name alias first: "(20s)" legacy presets map to distinct Long
    # presets, not to a duration variant of the short one.
    stripped_full = _strip_emoji(raw)
    for old in sorted(aliases, key=len, reverse=True):
        if stripped_full == _strip_emoji(old):
            return aliases[old], None
    duration = None
    match = _DURATION_SUFFIX.search(raw)
    if match:
        duration = f"{match.group(1)}s"
        raw = raw[:match.start()].strip()
    stripped = _strip_emoji(raw)
    for old in sorted(aliases, key=len, reverse=True):
        if stripped == _strip_emoji(old):
            return aliases[old], duration
    return stripped, duration


def resolve_prompt_key(name, duration, flat, durations, aliases):
    """Resolve a widget preset + duration selection to a flat prompt key.
    Alias-derived durations win over the widget default."""
    base, alias_dur = resolve_preset(name, aliases)
    dur = alias_dur or duration or durations.get(base) or DEFAULT_DURATION
    key = f"{base} ({dur})"
    if key in flat:
        return base, key
    if base in flat:
        return base, base
    fallback_dur = durations.get(base)
    if fallback_dur and f"{base} ({fallback_dur})" in flat:
        return base, f"{base} ({fallback_dur})"
    return base, name  # unknown preset: return raw (used as inline template)


def resolve_vl_preset(name, duration=None):
    return resolve_prompt_key(name, duration, VL_PROMPTS, VL_DURATIONS, VL_ALIASES)


def resolve_text_style(name, duration=None):
    return resolve_prompt_key(name, duration, TEXT_PROMPTS, TEXT_DURATIONS, TEXT_ALIASES)
