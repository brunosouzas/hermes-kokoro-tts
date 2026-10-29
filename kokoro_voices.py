"""Kokoro voice catalog helpers shared by the Hermes plugin and the local server.

Kokoro voice ids look like ``af_heart``: the first letter is the language, the
second the speaker's gender (``f``/``m``), and the rest the voice name. This
module is pure standard library so both sides can import it.
"""

from __future__ import annotations

import re
from typing import Optional

# Kokoro-82M v1.0 language codes (first letter of a voice id).
LANGUAGES: dict[str, tuple[str, str]] = {
    "a": ("en-us", "American English"),
    "b": ("en-gb", "British English"),
    "e": ("es", "Spanish"),
    "f": ("fr", "French"),
    "h": ("hi", "Hindi"),
    "i": ("it", "Italian"),
    "j": ("ja", "Japanese"),
    "p": ("pt-br", "Brazilian Portuguese"),
    "z": ("zh", "Mandarin Chinese"),
}

DEFAULT_VOICE = "af_heart"
VOICE_ID = re.compile(r"^[a-z][fm]_[a-z0-9]+$")


def is_valid_voice_id(voice: object) -> bool:
    """True for a well-formed Kokoro voice id. Rejects paths, so a voice can never
    be used to read an arbitrary file."""
    return isinstance(voice, str) and bool(VOICE_ID.match(voice)) and voice[0] in LANGUAGES


def voice_map(raw: object, languages: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
    """``settings.voices`` as ``(usable, problems)``: the language -> voice entries a sentence can
    reach, and one message per entry that is ignored (a language the plugin does not detect, or
    a voice that is not a Kokoro voice id), so nothing is dropped without a word."""
    if raw is None or raw == {}:
        return {}, []
    if not isinstance(raw, dict):
        return {}, ["voices must be a map from a language to a voice, e.g. {pt: pf_dora, en: bf_emma}; ignored"]
    usable: dict[str, str] = {}
    problems: list[str] = []
    for key, voice in raw.items():
        lang = str(key).strip().lower()
        if lang not in languages:
            problems.append(f"voices.{lang} ({voice}) is ignored: sentences are detected as "
                            f"{', '.join(languages)} only")
        elif not is_valid_voice_id(voice):
            problems.append(f"voices.{lang}: {voice!r} is not a Kokoro voice id and is ignored")
        elif lang in usable:
            problems.append(f"voices.{lang} is set twice; using {voice}, not {usable[lang]}")
            usable[lang] = voice
        else:
            usable[lang] = voice
    return usable, problems


def lang_code_for(voice: str, override: Optional[str] = None) -> str:
    """Kokoro language code for ``voice``; ``override`` wins when it is a known code."""
    if override:
        code = override.strip().lower()
        if code in LANGUAGES:
            return code
        for key, (bcp47, _label) in LANGUAGES.items():
            if code == bcp47:
                return key
        raise ValueError(f"Unknown Kokoro language {override!r}; use one of {sorted(LANGUAGES)}")
    if not is_valid_voice_id(voice):
        raise ValueError(f"Not a Kokoro voice id: {voice!r} (expected something like 'af_heart')")
    return voice[0]


def describe(voice: str) -> dict[str, str]:
    """Catalog entry for ``hermes setup`` style pickers."""
    code = voice[0]
    bcp47, label = LANGUAGES.get(code, ("", "Unknown"))
    gender = {"f": "female", "m": "male"}.get(voice[1], "")
    name = voice.split("_", 1)[1].replace("_", " ").title()
    return {"id": voice, "display": f"{name} ({label})", "language": bcp47, "gender": gender}
