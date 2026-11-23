"""Language helpers: names, ID3 codes, and detecting the language of a *text*.

Whisper decides the language of the *audio* (``WhisperModel.detect_language``); this
module covers everything else:

* :func:`language_name` - ``"ja"`` -> ``"Japanese"`` (the 99 languages Whisper knows)
* :func:`iso639_2` - the 3-letter code ID3's ``SYLT``/``USLT`` frames want
* :func:`detect_text_language` - script analysis (kana, hangul, Cyrillic ...), then the
  optional ``langdetect`` package, then a tiny stop-word model, so it works with no
  extra dependency at all
* :func:`decide_language` - how an audio guess and a lyrics guess are combined

Pure Python, no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian",
    "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian",
    "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian",
    "sw": "Swahili", "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans",
    "oc": "Occitan", "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi",
    "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek",
    "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen",
    "nn": "Nynorsk", "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish",
    "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
    "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa", "ba": "Bashkir",
    "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

# ID3v2 wants ISO 639-2 (bibliographic) codes; "XXX" is the spec's "unknown".
_ISO_639_2 = {
    "en": "eng", "zh": "chi", "de": "ger", "es": "spa", "ru": "rus", "ko": "kor", "fr": "fre",
    "ja": "jpn", "pt": "por", "tr": "tur", "pl": "pol", "ca": "cat", "nl": "dut", "ar": "ara",
    "sv": "swe", "it": "ita", "id": "ind", "hi": "hin", "fi": "fin", "vi": "vie", "he": "heb",
    "uk": "ukr", "el": "gre", "ms": "may", "cs": "cze", "ro": "rum", "da": "dan", "hu": "hun",
    "ta": "tam", "no": "nor", "th": "tha", "ur": "urd", "hr": "hrv", "bg": "bul", "lt": "lit",
    "la": "lat", "sk": "slo", "fa": "per", "lv": "lav", "bn": "ben", "sr": "srp", "sl": "slv",
    "et": "est", "mk": "mac", "is": "ice", "hy": "arm", "ka": "geo", "sq": "alb", "sw": "swa",
    "af": "afr", "cy": "wel", "eu": "baq", "gl": "glg", "tl": "tgl", "yue": "chi",
}


def language_name(code: Optional[str]) -> str:
    """``"ja"`` -> ``"Japanese"``; unknown codes are returned upper-cased."""
    code = (code or "").strip().lower()
    return LANGUAGE_NAMES.get(code, code.upper() or "Unknown")


def iso639_2(code: Optional[str]) -> str:
    """Three-letter language code for ID3 frames (``"XXX"`` when unknown)."""
    return _ISO_639_2.get((code or "").strip().lower(), "XXX")


@dataclass(frozen=True)
class LanguageGuess:
    """A language and how sure we are (0-1). ``source``: audio | lyrics | settings | audio+lyrics."""

    code: str
    probability: float
    source: str = "audio"

    @property
    def name(self) -> str:
        return language_name(self.code)

    def describe(self) -> str:
        return f"{self.name} ({self.probability:.0%})"


# ------------------------------------------------------------------ text analysis
_SCRIPT_RANGES = (
    ("kana", ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F))),
    ("han", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
    ("hangul", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("cyrillic", ((0x0400, 0x052F),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("devanagari", ((0x0900, 0x097F),)),
    ("greek", ((0x0370, 0x03FF),)),
)


def script_profile(text: str) -> dict[str, float]:
    """Share of each writing system among the *letters* of *text* (they sum to 1)."""
    counts: dict[str, int] = {}
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        cp = ord(ch)
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            key = "latin" if unicodedata.name(ch, "").startswith("LATIN") else "other"
            counts[key] = counts.get(key, 0) + 1
    return {name: n / letters for name, n in counts.items()} if letters else {}


_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and you i to a of in it is that my me we your for on with this be are but "
                    "so not all just oh like love baby when what how".split()),
    "es": frozenset("el la los las de que y en un una es no te me mi tu por con para lo se su al "
                    "como pero más mas yo si ya muy amor corazón".split()),
    "fr": frozenset("le la les de des et un une est que qui pas je tu il elle nous vous ce dans "
                    "pour sur avec mon ma mes ton au aux plus mais ne se cest jai".split()),
    "de": frozenset("der die das und ich du nicht ist ein eine zu den dem mit sie wir ihr es auf "
                    "für von mein dein auch aber wie wenn noch nur".split()),
    "it": frozenset("il lo la le gli di che e un una non è sei mi ti ci per con sono io tu lui "
                    "lei noi voi come ma più cuore amore mio mia".split()),
    "pt": frozenset("o a os as de que e um uma não nao é eu tu ele ela nós com para por em do da "
                    "meu minha você voce mas mais amor coração".split()),
    "nl": frozenset("de het een en ik je jij niet is dat die dit van op te voor met zijn ze we "
                    "maar ook wat nog".split()),
    "id": frozenset("yang dan di ke dari ini itu aku kamu dia kita tidak untuk dengan pada ada "
                    "saya cinta hati bukan akan masih sudah".split()),
    "tr": frozenset("bir ve bu da de ben sen o ne için ile çok değil gibi ama mi var yok sev "
                    "aşk kalbim".split()),
}
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_CYRILLIC_UNIQUE = {"uk": set("іїєґ"), "bg": set(), "sr": set("ђћџ"), "be": set("ў")}


def _langdetect(text: str) -> Optional[tuple[str, float]]:
    """Top guess from the optional ``langdetect`` package, or None when unavailable."""
    try:
        from langdetect import DetectorFactory, detect_langs  # type: ignore
        from langdetect.lang_detect_exception import LangDetectException  # type: ignore
    except ImportError:
        return None
    try:
        DetectorFactory.seed = 0          # deterministic
        guesses = detect_langs(text)
    except LangDetectException:
        return None
    except Exception:
        return None
    if not guesses:
        return None
    best = guesses[0]
    code = best.lang.lower()
    code = {"zh-cn": "zh", "zh-tw": "zh"}.get(code, code)
    return code, float(best.prob)


def _stopword_guess(text: str) -> Optional[tuple[str, float]]:
    words = [w.casefold() for w in _WORD_RE.findall(text)]
    if len(words) < 8:
        return None
    scores = {code: sum(1 for w in words if w in vocab) / len(words)
              for code, vocab in _STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < 0.12:
        return None
    confidence = max(0.0, min(0.9, 0.45 + (top - second) * 3.0))
    return (best, confidence) if confidence >= 0.5 else None


def detect_text_language(text: str) -> Optional[LanguageGuess]:
    """Best guess of the language *text* is written in, or ``None`` when it cannot tell.

    Kana, hangul, Thai, Hebrew, Greek and Devanagari identify a language by themselves;
    Han-only text is Chinese; Cyrillic/Arabic/Latin text goes through ``langdetect`` when
    installed, else through a small stop-word model (so it works with no extra package).
    """
    text = (text or "").strip()
    if len(text) < 12:
        return None
    prof = script_profile(text)
    if not prof:
        return None

    def guess(code: str, p: float) -> LanguageGuess:
        return LanguageGuess(code, p, "lyrics")

    if prof.get("kana", 0) >= 0.08:
        return guess("ja", 0.97)
    if prof.get("hangul", 0) >= 0.30:
        return guess("ko", 0.97)
    if prof.get("han", 0) >= 0.50:
        return guess("zh", 0.90)
    if prof.get("thai", 0) >= 0.50:
        return guess("th", 0.97)
    if prof.get("hebrew", 0) >= 0.50:
        return guess("he", 0.95)
    if prof.get("greek", 0) >= 0.50:
        return guess("el", 0.95)
    if prof.get("devanagari", 0) >= 0.50:
        refined = _langdetect(text)
        return guess(refined[0], refined[1]) if refined and refined[0] in ("hi", "mr", "ne") else guess("hi", 0.80)
    dominant = max(prof, key=prof.get)
    share = prof[dominant]
    if share < 0.60:
        return None
    if dominant == "cyrillic":
        refined = _langdetect(text)
        if refined and refined[0] in ("ru", "uk", "bg", "sr", "mk", "be", "kk"):
            return guess(*refined)
        letters = set(text.casefold())
        for code, unique in _CYRILLIC_UNIQUE.items():
            if unique & letters:
                return guess(code, 0.80)
        return guess("ru", 0.70)
    if dominant == "arabic":
        refined = _langdetect(text)
        if refined and refined[0] in ("ar", "fa", "ur", "ps", "sd"):
            return guess(*refined)
        return guess("ar", 0.70)
    if dominant == "latin":
        refined = _langdetect(text)
        if refined and refined[1] >= 0.60:
            return guess(*refined)
        fallback = _stopword_guess(text)
        return guess(*fallback) if fallback else None
    return None


# --------------------------------------------------------------------- the decision
CONFIDENT = 0.60          # audio guess we trust on its own
USABLE = 0.35             # below this the audio guess is noise
TEXT_TRUSTED = 0.80       # lyrics guess strong enough to overrule a shaky audio guess


def decide_language(audio: Optional[LanguageGuess],
                    text: Optional[LanguageGuess]) -> tuple[Optional[LanguageGuess], str]:
    """Pick the language Whisper should be locked to. Returns ``(guess, note)``.

    *What is physically sung* wins. The lyrics only break ties and rescue a shaky audio
    guess (instrumental intro, heavy effects) - except that they never override a
    confident audio result; a disagreement is reported in ``note`` instead.
    """
    if audio is None and text is None:
        return None, ""
    if audio is None:
        return (text, "from the lyrics") if text and text.probability >= TEXT_TRUSTED else (None, "")
    if text is None:
        return (audio, "") if audio.probability >= USABLE else (None, "too uncertain to lock")
    if audio.code == text.code:
        merged = LanguageGuess(audio.code, max(audio.probability, min(0.99, (audio.probability + text.probability) / 2 + 0.05)),
                               "audio+lyrics")
        return merged, "audio and lyrics agree"
    if audio.probability >= CONFIDENT:
        return audio, f"lyrics look {text.name}, but the audio sounds {audio.name}"
    if text.probability >= TEXT_TRUSTED:
        return LanguageGuess(text.code, text.probability, "lyrics"), \
            f"audio guess ({audio.describe()}) was shaky; using the lyrics' language"
    return (audio, "") if audio.probability >= USABLE else (None, "too uncertain to lock")
