"""English words inside non-English sentences, pronounced in English by the same voice.

Kokoro speaks from phonemes. For a sentence that contains words from the user's dictionary, the
whole sentence is phonemized with the voice's own language (so stress and vowel quality of the
surrounding words stay natural) and only the phoneme words of the listed words are swapped for
English phonemes. Pure standard library: the two phonemizers are passed in.

Alignment is checked twice before anything is swapped: the sentence must yield one phoneme word
per text word, and at every swapped position the phoneme word must resemble the native
pronunciation of that very word. Anything else falls back to normal synthesis.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Iterable, Optional

G2P = Callable[[str], str]

# Words as espeak counts them: letters with inner hyphens or apostrophes ("e-mail", "d'água"),
# or numbers with inner separators ("4.9", "12:30"), which espeak also reads as one word.
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|\d+(?:[.,:]\d+)*", re.UNICODE)
# Punctuation espeak keeps attached to a phoneme word (it is prosody, so it must survive a swap).
_LEAD = re.compile(r"^[(\[{\"'“‘«¿¡]+")
_TRAIL = re.compile(r"[.,!?;:…)\]}\"'”’»]+$")
_STRESS = re.compile(r"[ˈˌ]")
MAX_ENTRY_WORDS = 4
MAX_ENTRIES = 500
MIN_SIMILARITY = 0.5


def entry_words(entry: object) -> Optional[tuple[str, ...]]:
    """The lowercase words of a dictionary entry, or ``None`` if it cannot be an entry."""
    if not isinstance(entry, str):
        return None
    words = tuple(w.lower() for w in _WORD.findall(entry))
    if not words or len(words) > MAX_ENTRY_WORDS or any(w[0].isdigit() for w in words):
        return None
    return words


def normalize(entries: Iterable[object]) -> list[tuple[str, ...]]:
    """Dictionary entries as lowercase word tuples, longest first ("pull request" before "pull")."""
    seen: dict[tuple[str, ...], None] = {}
    for entry in entries:
        words = entry_words(entry)
        if words is None:
            continue
        seen.setdefault(words, None)
        if len(seen) >= MAX_ENTRIES:
            break
    return sorted(seen, key=len, reverse=True)


def _matches(words: list[str], start: int, entry: tuple[str, ...]) -> bool:
    if start + len(entry) > len(words):
        return False
    for offset, expected in enumerate(entry):
        word = words[start + offset].lower()
        last = offset == len(entry) - 1
        # A plural of the last word counts too: "cards" for "card", "pull requests".
        if word != expected and not (last and word == expected + "s"):
            return False
    return True


def find(text: str, entries: list[tuple[str, ...]]) -> list[tuple[int, int]]:
    """``(first_word, end_word)`` index spans of dictionary hits, non-overlapping. A phrase only
    matches when its words are separated by plain whitespace, never across punctuation."""
    found = list(_WORD.finditer(text))
    words = [m.group(0) for m in found]
    gaps = [text[found[i].end():found[i + 1].start()] for i in range(len(found) - 1)]
    spans, index = [], 0
    while index < len(words):
        for entry in entries:
            if _matches(words, index, entry) and all(
                    gaps[index + k].isspace() for k in range(len(entry) - 1)):
                spans.append((index, index + len(entry)))
                index += len(entry)
                break
        else:
            index += 1
    return spans


def _core(token: str) -> str:
    return _STRESS.sub("", _TRAIL.sub("", _LEAD.sub("", token)))


def _aligned(token: str, word: str, native: G2P) -> bool:
    """Does this phoneme word plausibly belong to ``word``? Context changes details (a final
    /s/ voices to /z/ before a vowel, stress moves), so the test is similarity, not equality."""
    alone = _core(native(word.lower()).strip())
    here = _core(token)
    return bool(alone) and difflib.SequenceMatcher(None, alone, here).ratio() >= MIN_SIMILARITY


def splice(text: str, entries: list[tuple[str, ...]], native: G2P, english: G2P) -> Optional[str]:
    """Phonemes for ``text`` with the listed words in English, or ``None`` when no listed word
    occurs or the phonemes cannot be lined up with the words (the caller then synthesizes the
    text normally)."""
    spans = find(text, entries)
    if not spans:
        return None
    words = _WORD.findall(text)
    tokens = native(text).strip().split()
    if len(tokens) != len(words):
        return None
    for start, end in spans:
        if not all(_aligned(tokens[i], words[i], native) for i in range(start, end)):
            return None
    out: list[str] = []
    index = 0
    for start, end in spans:
        out.extend(tokens[index:start])
        # Lowercase: the lexicon is case-insensitive for ordinary words, and a capital at the
        # start of a sentence must not turn "Deploy" into a different lookup.
        spoken = english(" ".join(words[start:end]).lower()).strip()
        if not spoken or "❓" in spoken or len(spoken.split()) != end - start:
            out.extend(tokens[start:end])  # no usable English pronunciation: keep the native one
        else:
            lead = _LEAD.match(tokens[start])
            trail = _TRAIL.search(tokens[end - 1])
            out.append((lead.group(0) if lead else "") + spoken + (trail.group(0) if trail else ""))
        index = end
    out.extend(tokens[index:])
    return " ".join(out)
