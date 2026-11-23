"""Two-tier, search-first lyrics chain.

Tier 1 - SYNCED providers (a finished ``.lrc`` that can be saved as-is)
    1. LRCLIB         open database, JSON API
    2. BetterLyrics   Apple-Music style TTML converted to LRC
    3. RCLyricsBand   LRC site, HTML scraping
    4. Megalobiz      LRC site, HTML scraping
    5. NetEase        public web API (huge catalogue, great for CJK + pop)
    6. QQ Music       public web API

Tier 2 - PLAIN TEXT providers (official words, no timing)
    7. Genius         official API when a token is set, public search + page scrape otherwise
    8. Lyrics.ovh     free JSON API (Deezer-backed suggestions)
    9. Musixmatch     official API, only when a key is configured (free tier = excerpt)

Every hit is returned as a :class:`LyricsResult` whose ``kind`` says what to do next:
``SYNCED`` -> save it; ``PLAIN_TEXT`` -> hand it to the forced aligner.

Design rules
------------
* Queries are sanitised first (``text_utils.query_variants``).
* Every candidate is fuzzy-scored (artist/title/duration) and validated (does the
  synced text fit the track? does the plain text look like lyrics?).
* A failing provider (offline, 5xx, rate limit, layout change) is logged and skipped;
  it can never abort the chain or the batch.
* Network calls have timeouts, bounded retries and per-provider throttling.

Honest caveat: NetEase, QQ Music, RCLyricsBand, Megalobiz and Genius are unofficial
web endpoints/scrapes. They change without notice; each provider is isolated so a
broken one only costs you that provider.
"""

from __future__ import annotations

import base64
import binascii
import html
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import quote, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .lrc_utils import (
    build_lrc, clean_plain_lyrics, drop_credit_lines, looks_like_lyrics, parse_lrc,
    parse_ttml, strip_to_plain, validate_synced,
)
from .models import IdentifiedSong, LyricsResult, MatchKind, TrackInfo
from .text_utils import query_variants, similarity

logger = logging.getLogger(__name__)

USER_AGENT = "Lyricist/3.1 (desktop lyric sync tool; formerly Local Sync Lyrics)"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
LogFn = Callable[[str, str, str], None]  # (tag, message, level)

TAG_T1 = "API: Tier 1"
TAG_PLAIN = "API: Plain Text"
_TRANSLATION_RE = re.compile(
    r"(?:translation|traduc|romaniz|übersetzung|tradução|çeviri|перевод|traduzione)",
    re.IGNORECASE)


class ProviderError(Exception):
    """A provider could not be queried (network, HTTP status, bad payload)."""


# ---------------------------------------------------------------------- query
@dataclass
class TrackQuery:
    variants: list[tuple[str, str]]
    album: str = ""
    duration: Optional[float] = None

    @classmethod
    def from_track(cls, track: TrackInfo) -> "TrackQuery":
        return cls(variants=query_variants(track.artist, track.title),
                   album=track.album, duration=track.duration)

    @classmethod
    def from_names(cls, artist: str, title: str, duration: Optional[float] = None) -> "TrackQuery":
        return cls(variants=query_variants(artist, title), duration=duration)

    @property
    def primary(self) -> tuple[str, str]:
        return self.variants[0] if self.variants else ("", "")


def score_candidate(query: TrackQuery, cand_artist: str, cand_title: str,
                    cand_duration: Optional[float]) -> float:
    """0-100 confidence that a candidate is the track we are looking for."""
    cand_artist = str(cand_artist or "")
    cand_title = str(cand_title or "")
    best = 0.0
    for q_artist, q_title in query.variants:
        title_score = similarity(q_title, cand_title)
        if q_artist and cand_artist:
            score = 0.65 * title_score + 0.35 * similarity(q_artist, cand_artist)
        elif q_artist and not cand_artist:
            score = title_score * 0.85
        else:  # title-only query: demand an excellent title match
            score = title_score * 0.9
        best = max(best, score)

    if query.duration and cand_duration:
        diff = abs(query.duration - cand_duration)
        if diff <= 3:
            best += 8
        elif diff <= 8:
            best += 2
        elif diff <= 20:
            best -= 12
        else:
            best -= 100  # different recording
    return max(0.0, min(100.0, best))


# ----------------------------------------------------------------------- HTTP
class HttpClient:
    """requests.Session with retries, timeouts and a friendly User-Agent."""

    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(
            total=2, backoff_factor=0.7, status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "POST"), respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=6, pool_maxsize=6)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, */*"})

    @staticmethod
    def _check(response: requests.Response) -> Optional[requests.Response]:
        if response.status_code in (404, 400, 204, 410):
            return None
        if response.status_code in (401, 403):
            raise ProviderError(f"access denied (HTTP {response.status_code})")
        if response.status_code == 429:
            raise ProviderError("rate limited (HTTP 429)")
        if response.status_code >= 400:
            raise ProviderError(f"HTTP {response.status_code}")
        return response

    def get(self, url: str, params: Optional[dict] = None,
            headers: Optional[dict] = None) -> Optional[requests.Response]:
        """GET -> Response, or ``None`` for 'no such thing' (404/400/204).

        Raises :class:`ProviderError` for transport problems / rate limits / 5xx.
        """
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.Timeout as exc:
            raise ProviderError("request timed out") from exc
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "?")
            raise ProviderError(f"HTTP {status}") from exc
        except requests.RequestException as exc:
            raise ProviderError(f"network error: {exc.__class__.__name__}") from exc
        return self._check(response)

    def post(self, url: str, data: Optional[dict] = None,
             headers: Optional[dict] = None) -> Optional[requests.Response]:
        try:
            response = self.session.post(url, data=data, headers=headers, timeout=self.timeout)
        except requests.Timeout as exc:
            raise ProviderError("request timed out") from exc
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "?")
            raise ProviderError(f"HTTP {status}") from exc
        except requests.RequestException as exc:
            raise ProviderError(f"network error: {exc.__class__.__name__}") from exc
        return self._check(response)

    def get_json(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
        response = self.get(url, params, headers)
        if response is None:
            return None
        return _json_of(response)

    def close(self) -> None:
        self.session.close()


def _json_of(response):
    """Parse JSON, tolerating JSONP wrappers like ``callback({...})``."""
    try:
        return response.json()
    except ValueError:
        pass
    match = re.search(r"^\s*[\w$.]+\s*\((.*)\)\s*;?\s*$", response.text or "", re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except ValueError:
            pass
    raise ProviderError("invalid JSON in response")


def _as_dict(value) -> dict:
    """Strict type guard: providers sometimes return a string where JSON was expected."""
    return value if isinstance(value, dict) else {}


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _as_text(value) -> str:
    return value if isinstance(value, str) else ""


def _dig(value, *keys):
    """``_dig(payload, "result", "songs")`` - safe nested lookup (never raises)."""
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _soup(text: str):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ProviderError("beautifulsoup4 is not installed") from exc
    return BeautifulSoup(text, "html.parser")


# ------------------------------------------------------------------ providers
class LyricsProvider(ABC):
    id = "base"
    label = "Base"
    tier = 1
    min_interval = 0.25  # seconds between requests (politeness)

    def __init__(self, http: HttpClient, settings: Settings) -> None:
        self.http = http
        self.settings = settings
        self._last_call = 0.0

    def available(self) -> bool:
        """False when a required credential is missing (provider is skipped)."""
        return True

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @abstractmethod
    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        """Return the best hit or ``None``. May raise :class:`ProviderError`."""

    def identify(self, snippet: str) -> list[IdentifiedSong]:
        """Find songs whose lyrics contain *snippet* (optional capability)."""
        return []

    # shared helper ---------------------------------------------------------
    def _synced_result(self, lrc_text: str, query: TrackQuery, artist: str, title: str,
                       score: float, *, strip_credits: bool = False) -> Optional[LyricsResult]:
        lines = parse_lrc(_as_text(lrc_text))
        if strip_credits:
            lines = drop_credit_lines(lines)
        if not lines:
            return None
        lrc = build_lrc(lines, artist=artist, title=title, album=query.album,
                        duration=query.duration, source=self.label)
        ok, reason, _ = validate_synced(lrc, query.duration)
        if not ok:
            logger.debug("%s candidate rejected: %s", self.label, reason)
            return None
        return LyricsResult(provider=self.label, synced_lrc=lrc, artist=artist, title=title,
                            score=score, tier=self.tier)


# ================================================================== TIER 1
class LrclibProvider(LyricsProvider):
    """https://lrclib.net - open database of synced lyrics (best first choice)."""

    id, label = "lrclib", "LRCLIB"
    BASE = "https://lrclib.net/api"

    def _to_result(self, item: dict, query: TrackQuery) -> Optional[LyricsResult]:
        if not isinstance(item, dict):
            return None
        artist, title = _as_text(item.get("artistName")), _as_text(item.get("trackName"))
        duration = item.get("duration")
        score = score_candidate(
            query, artist, title, float(duration) if isinstance(duration, (int, float)) else None)
        plain = clean_plain_lyrics(_as_text(item.get("plainLyrics"))) or None
        synced = _as_text(item.get("syncedLyrics")) or None
        return LyricsResult(
            provider=self.label, synced_lrc=synced, plain=plain,
            instrumental=bool(item.get("instrumental")), artist=artist, title=title,
            score=score, tier=1,
        )

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        threshold = self.settings.min_match_score
        candidates: list[LyricsResult] = []

        # 1) exact signature lookup (fast, most precise)
        artist, title = query.primary
        if artist and title:
            params = {"artist_name": artist, "track_name": title}
            if query.album:
                params["album_name"] = query.album
            if query.duration:
                params["duration"] = int(round(query.duration))
            self._throttle()
            data = self.http.get_json(f"{self.BASE}/get", params)
            if isinstance(data, dict):
                hit = self._to_result(data, query)
                if hit:
                    candidates.append(hit)

        # 2) fuzzy search over the sanitised variants
        if not any(c.has_synced and c.score >= threshold for c in candidates):
            for q_artist, q_title in query.variants[:3]:
                params = {"track_name": q_title}
                if q_artist:
                    params["artist_name"] = q_artist
                self._throttle()
                data = self.http.get_json(f"{self.BASE}/search", params)
                for item in data if isinstance(data, list) else []:
                    hit = self._to_result(item, query)
                    if hit:
                        candidates.append(hit)
                if any(c.has_synced and c.score >= threshold + 10 for c in candidates):
                    break
            if not candidates:  # free-text fallback
                q_artist, q_title = query.primary
                self._throttle()
                data = self.http.get_json(f"{self.BASE}/search", {"q": f"{q_artist} {q_title}".strip()})
                for item in data if isinstance(data, list) else []:
                    hit = self._to_result(item, query)
                    if hit:
                        candidates.append(hit)

        good = [c for c in candidates if c.score >= threshold]
        if not good:
            return None
        synced = sorted((c for c in good if c.has_synced), key=lambda c: -c.score)
        if synced:
            return synced[0]
        instrumental = [c for c in good if c.instrumental]
        if instrumental:
            return max(instrumental, key=lambda c: c.score)
        plain = [c for c in good if c.plain]
        return max(plain, key=lambda c: c.score) if plain else None


class BetterLyricsProvider(LyricsProvider):
    """https://lyrics-api.boidu.dev - Apple-Music style TTML converted to LRC."""

    id, label = "betterlyrics", "BetterLyrics"
    BASE = "https://lyrics-api.boidu.dev"
    min_interval = 0.4

    @staticmethod
    def _find_ttml(payload) -> str:
        """Locate a TTML document anywhere inside a JSON payload."""
        if isinstance(payload, str):
            probe = payload.lstrip()[:200].lower()
            return payload if ("<tt" in probe or "<?xml" in probe or "<p " in probe) else ""
        if isinstance(payload, dict):
            for key in ("ttml", "lyrics", "data", "result"):
                if key in payload:
                    found = BetterLyricsProvider._find_ttml(payload[key])
                    if found:
                        return found
            for value in payload.values():
                found = BetterLyricsProvider._find_ttml(value)
                if found:
                    return found
        if isinstance(payload, list):
            for value in payload:
                found = BetterLyricsProvider._find_ttml(value)
                if found:
                    return found
        return ""

    def _fetch(self, path: str, artist: str, title: str, duration: Optional[float]) -> str:
        params = {"s": title, "a": artist}
        if duration:
            params["d"] = int(round(duration))
        self._throttle()
        response = self.http.get(f"{self.BASE}{path}", params)
        if response is None:
            return ""
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return self._find_ttml(payload)

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        tried: set[tuple[str, str]] = set()
        for artist, title in query.variants[:2]:
            if not artist or (artist, title) in tried:
                continue  # the API needs both artist and song
            tried.add((artist, title))
            ttml = self._fetch("/getLyrics", artist, title, query.duration)
            if not ttml:
                ttml = self._fetch("/ttml/getLyrics", artist, title, query.duration)
            if not ttml:
                continue
            lines = parse_ttml(ttml)
            if not lines:
                continue
            lrc = build_lrc(lines, artist=artist, title=title, album=query.album,
                            duration=query.duration, source=self.label)
            ok, reason, _ = validate_synced(lrc, query.duration)
            if not ok:
                logger.debug("BetterLyrics candidate rejected: %s", reason)
                continue
            # The API does not echo the matched title, so confidence comes from the
            # duration check above; report a neutral-high score.
            return LyricsResult(provider=self.label, synced_lrc=lrc, artist=artist,
                                title=title, score=80.0, tier=1)
        return None


class RcLyricsBandProvider(LyricsProvider):
    """https://rclyricsband.com - community LRC files (form search + HTML scrape).

    Flow (mirrors the long-standing MythTV/CU-LRC scraper): POST ``search=<artist
    title>`` to the home page, read the ``song_search`` links, open the best one and
    read the block marked ``lrc_text_format``.
    """

    id, label = "rclyricsband", "RCLyricsBand"
    BASE = "https://rclyricsband.com/"
    min_interval = 0.8

    def _headers(self) -> dict:
        return {"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.8", "Referer": self.BASE}

    @staticmethod
    def _extract_lrc(page_html: str) -> str:
        soup = _soup(page_html)
        node = soup.find(class_="lrc_text_format")
        if node is not None:
            for br in node.find_all("br"):
                br.replace_with("\n")
            return node.get_text()
        match = re.search(r"lrc_text_format['\"]?>(.*?)</(?:p|div)>", page_html,
                          re.DOTALL | re.IGNORECASE)
        if match:
            return html.unescape(re.sub(r"<br\s*/?>", "\n", match.group(1)))
        return ""

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        artist, title = query.primary
        term = f"{artist} {title}".strip()
        if not term:
            return None
        self._throttle()
        response = self.http.post(self.BASE, {"search": term}, headers=self._headers())
        if response is None:
            return None
        ranked: list[tuple[float, str, str]] = []
        for anchor in _soup(response.text).find_all("a", class_="song_search"):
            href, text = anchor.get("href") or "", anchor.get_text(" ", strip=True)
            if href:
                ranked.append((similarity(term, text), href, text))   # "Title - Artist": order-agnostic
        ranked.sort(key=lambda r: -r[0])

        for score, href, _text in ranked[:3]:
            if score < self.settings.min_match_score - 10:
                break
            self._throttle()
            page = self.http.get(urljoin(self.BASE, href), headers=self._headers())
            if page is None:
                continue
            hit = self._synced_result(self._extract_lrc(page.text), query, artist, title, float(score))
            if hit:
                return hit
        return None


class MegalobizProvider(LyricsProvider):
    """https://www.megalobiz.com - community LRC files (HTML scraping)."""

    id, label = "megalobiz", "Megalobiz"
    BASE = "https://www.megalobiz.com"
    min_interval = 0.8

    def _headers(self) -> dict:
        return {"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.8"}

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        artist, title = query.primary
        term = f"{artist} {title}".strip()
        if not term:
            return None
        self._throttle()
        response = self.http.get(
            f"{self.BASE}/search/all",
            {"qry": term, "searchButton.x": 0, "searchButton.y": 0}, headers=self._headers())
        if response is None:
            return None
        ranked: list[tuple[float, str, str]] = []
        for anchor in _soup(response.text).find_all("a", class_="entity_name"):
            href = anchor.get("href") or ""
            if href.startswith("/lrc/maker/"):
                text = anchor.get_text(" ", strip=True)
                ranked.append((similarity(term, text), href, text))
        ranked.sort(key=lambda r: -r[0])

        for score, href, _text in ranked[:3]:
            if score < self.settings.min_match_score - 10:
                break
            self._throttle()
            page = self.http.get(f"{self.BASE}{href}", headers=self._headers())
            if page is None:
                continue
            details = _soup(page.text).find("div", class_="lyrics_details")
            if details is None:
                continue
            hit = self._synced_result(details.get_text("\n"), query, artist, title, float(score))
            if hit:
                return hit
        return None


class NetEaseProvider(LyricsProvider):
    """NetEase Cloud Music public web endpoints (search + lyric)."""

    id, label = "netease", "NetEase"
    SEARCH = "https://music.163.com/api/search/get/web"
    LYRIC = "https://music.163.com/api/song/lyric"
    min_interval = 0.5

    def _headers(self) -> dict:
        return {"User-Agent": BROWSER_UA, "Referer": "https://music.163.com/",
                "Cookie": "appver=2.0.2; os=pc", "Accept": "application/json, text/plain, */*"}

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        artist, title = query.primary
        term = f"{artist} {title}".strip()
        if not term:
            return None
        self._throttle()
        data = self.http.get_json(
            self.SEARCH, {"s": term, "type": 1, "limit": 6, "offset": 0, "total": "true",
                          "csrf_token": ""}, headers=self._headers())
        # NetEase sometimes answers with a plain string / HTML error page.  Guard
        # every level instead of assuming a dict (the old `.get` crash).
        songs = _as_list(_dig(data, "result", "songs"))
        ranked: list[tuple[float, dict]] = []
        for song in songs:
            if not isinstance(song, dict) or not song.get("id"):
                continue
            artists = song.get("artists") or song.get("ar") or []
            if not isinstance(artists, list):
                artists = []
            cand_artist = ", ".join(
                _as_text(a.get("name")) for a in artists if isinstance(a, dict))
            dur = song.get("duration") or song.get("dt")
            seconds = float(dur) / 1000.0 if isinstance(dur, (int, float)) and dur > 1000 else None
            name = _as_text(song.get("name"))
            score = score_candidate(query, cand_artist, name, seconds)
            ranked.append((score, {**song, "_artist": cand_artist, "_name": name}))
        ranked.sort(key=lambda r: -r[0])

        for score, song in ranked[:2]:
            if score < self.settings.min_match_score:
                break
            self._throttle()
            lyric = self.http.get_json(
                self.LYRIC, {"id": song["id"], "lv": 1, "kv": 1, "tv": -1}, headers=self._headers())
            if not isinstance(lyric, dict) or lyric.get("nolyric") or lyric.get("uncollected"):
                continue
            text = _as_text(_dig(lyric, "lrc", "lyric"))
            hit = self._synced_result(text, query, song["_artist"] or artist,
                                      song.get("_name") or title, score, strip_credits=True)
            if hit:
                return hit
        return None


class QQMusicProvider(LyricsProvider):
    """QQ Music public web endpoints (search + lyric)."""

    id, label = "qqmusic", "QQ Music"
    SEARCH = "https://c.y.qq.com/soso/fcgi-bin/client_search_cp"
    LYRIC = "https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg"
    min_interval = 0.5

    def _headers(self) -> dict:
        return {"User-Agent": BROWSER_UA, "Referer": "https://y.qq.com/",
                "Accept": "application/json, text/plain, */*"}

    @staticmethod
    def _decode_lyric(raw: str) -> str:
        raw = (raw or "").strip()
        if not raw:
            return ""
        if "[" not in raw[:40]:                      # base64 payload (nobase64 ignored)
            try:
                raw = base64.b64decode(raw, validate=False).decode("utf-8", "replace")
            except (binascii.Error, ValueError):
                return ""
        return html.unescape(raw).replace("&#10;", "\n").replace("&apos;", "'")

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        artist, title = query.primary
        term = f"{artist} {title}".strip()
        if not term:
            return None
        self._throttle()
        data = self.http.get_json(
            self.SEARCH, {"w": term, "format": "json", "p": 1, "n": 6, "cr": 1, "g_tk": 5381,
                          "t": 0, "aggr": 1, "lossless": 0, "flag_qc": 0,
                          "inCharset": "utf8", "outCharset": "utf-8"},
            headers=self._headers())
        songs = (((data or {}).get("data") or {}).get("song") or {}).get("list") or []
        ranked: list[tuple[float, dict, str]] = []
        for song in songs:
            if not isinstance(song, dict):
                continue
            mid = song.get("songmid") or song.get("mid")
            name = song.get("songname") or song.get("name") or song.get("title") or ""
            singers = song.get("singer") or []
            cand_artist = ", ".join(s.get("name", "") for s in singers if isinstance(s, dict))
            interval = song.get("interval")
            seconds = float(interval) if isinstance(interval, (int, float)) and interval > 0 else None
            if mid:
                ranked.append((score_candidate(query, cand_artist, name, seconds),
                               {"mid": mid, "name": name}, cand_artist))
        ranked.sort(key=lambda r: -r[0])

        for score, song, cand_artist in ranked[:2]:
            if score < self.settings.min_match_score:
                break
            self._throttle()
            lyric = self.http.get_json(
                self.LYRIC, {"songmid": song["mid"], "format": "json", "nobase64": 1,
                             "g_tk": 5381, "inCharset": "utf8", "outCharset": "utf-8"},
                headers=self._headers())
            if not isinstance(lyric, dict):
                continue
            text = self._decode_lyric(lyric.get("lyric") or "")
            hit = self._synced_result(text, query, cand_artist or artist, song["name"] or title,
                                      score, strip_credits=True)
            if hit:
                return hit
        return None


# ================================================================== TIER 2
class GeniusProvider(LyricsProvider):
    """Genius: official search API when a token is set, public search otherwise.

    The API never returns lyrics, so the song page is fetched and the lyric
    containers are parsed. Genius sometimes blocks scrapers: that just skips it.
    """

    id, label, tier = "genius", "Genius", 2
    min_interval = 0.7

    def _api_headers(self) -> dict:
        token = self.settings.effective_genius_token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _hits(self, query_text: str) -> list[dict]:
        """Normalised ``{title, artist, url}`` song hits."""
        token = self.settings.effective_genius_token()
        self._throttle()
        if token:
            data = self.http.get_json("https://api.genius.com/search", {"q": query_text},
                                      headers=self._api_headers())
            raw_hits = (((data or {}).get("response") or {}).get("hits")) or []
        else:
            data = self.http.get_json("https://genius.com/api/search/multi",
                                      {"per_page": 5, "q": query_text},
                                      headers={"User-Agent": BROWSER_UA})
            raw_hits = []
            for section in (((data or {}).get("response") or {}).get("sections")) or []:
                if section.get("type") == "song":
                    raw_hits.extend(section.get("hits") or [])
        out: list[dict] = []
        for hit in raw_hits:
            result = (hit or {}).get("result") or {}
            title = result.get("title") or ""
            artist = (result.get("primary_artist") or {}).get("name") or ""
            url = result.get("url") or ""
            if not (url and title):
                continue
            if _TRANSLATION_RE.search(title) or _TRANSLATION_RE.search(artist):
                continue  # "Song (Traducción al Español)" pages are not the original
            out.append({"title": title, "artist": artist, "url": url})
        return out

    @staticmethod
    def _scrape(page_html: str) -> str:
        soup = _soup(page_html)
        parts: list[str] = []
        for container in soup.select('div[data-lyrics-container="true"]'):
            for junk in container.select('[data-exclude-from-selection="true"]'):
                junk.decompose()
            for br in container.find_all("br"):
                br.replace_with("\n")
            parts.append(container.get_text())
        return clean_plain_lyrics("\n".join(parts))

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        artist, title = query.primary
        term = f"{artist} {title}".strip()
        if not term:
            return None
        ranked = sorted(
            ((score_candidate(query, h["artist"], h["title"], None), h) for h in self._hits(term)),
            key=lambda r: -r[0])
        for score, hit in ranked[:2]:
            if score < self.settings.min_match_score:
                break
            self._throttle()
            page = self.http.get(hit["url"], headers={"User-Agent": BROWSER_UA,
                                                      "Accept": "text/html"})
            if page is None:
                continue
            text = self._scrape(page.text)
            if looks_like_lyrics(text):
                return LyricsResult(provider=self.label, plain=text, artist=hit["artist"],
                                    title=hit["title"], score=score, tier=2)
        return None

    def identify(self, snippet: str) -> list[IdentifiedSong]:
        return [IdentifiedSong(h["artist"], h["title"], self.label) for h in self._hits(snippet)[:4]]


class LyricsOvhProvider(LyricsProvider):
    """https://lyrics.ovh - free plain-text lyrics (suggest + lyrics endpoints)."""

    id, label, tier = "lyricsovh", "Lyrics.ovh", 2
    BASE = "https://api.lyrics.ovh"
    min_interval = 0.6

    def _lyrics(self, artist: str, title: str) -> str:
        self._throttle()
        data = self.http.get_json(f"{self.BASE}/v1/{quote(artist, safe='')}/{quote(title, safe='')}")
        text = data.get("lyrics") if isinstance(data, dict) else ""
        return clean_plain_lyrics(text or "")

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        for artist, title in query.variants[:3]:
            if not (artist and title):
                continue
            text = self._lyrics(artist, title)
            if looks_like_lyrics(text):
                return LyricsResult(provider=self.label, plain=text, artist=artist, title=title,
                                    score=85.0, tier=2)

        # Spelling/artist unknown: ask the suggestion index (Deezer-backed) first.
        artist, title = query.primary
        term = f"{artist} {title}".strip()
        if not term:
            return None
        self._throttle()
        data = self.http.get_json(f"{self.BASE}/suggest/{quote(term, safe='')}")
        ranked = []
        for item in (data.get("data") or []) if isinstance(data, dict) else []:
            cand_artist = ((item or {}).get("artist") or {}).get("name") or ""
            cand_title = (item or {}).get("title") or ""
            duration = (item or {}).get("duration")
            score = score_candidate(query, cand_artist, cand_title,
                                    float(duration) if isinstance(duration, (int, float)) else None)
            ranked.append((score, cand_artist, cand_title))
        ranked.sort(key=lambda r: -r[0])
        for score, cand_artist, cand_title in ranked[:2]:
            if score < self.settings.min_match_score:
                break
            text = self._lyrics(cand_artist, cand_title)
            if looks_like_lyrics(text):
                return LyricsResult(provider=self.label, plain=text, artist=cand_artist,
                                    title=cand_title, score=score, tier=2)
        return None


class MusixmatchProvider(LyricsProvider):
    """Musixmatch official API. Needs ``musixmatch_key``; the free plan only returns
    an excerpt, so results are flagged ``partial`` and used only as a last resort
    (and for identifying songs by a lyric snippet, which works on the free plan)."""

    id, label, tier = "musixmatch", "Musixmatch", 2
    BASE = "https://api.musixmatch.com/ws/1.1"
    min_interval = 0.6

    def available(self) -> bool:
        return bool(self.settings.effective_musixmatch_key())

    def _call(self, method: str, **params):
        params["apikey"] = self.settings.effective_musixmatch_key()
        self._throttle()
        data = self.http.get_json(f"{self.BASE}/{method}", params)
        message = (data or {}).get("message") or {}
        status = (message.get("header") or {}).get("status_code")
        if status in (401, 402):
            raise ProviderError("invalid or over-quota API key")
        if status != 200:
            return None
        return message.get("body")

    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        artist, title = query.primary
        body = self._call("matcher.lyrics.get", q_track=title, q_artist=artist)
        lyrics = (body.get("lyrics") or {}) if isinstance(body, dict) else {}
        text = clean_plain_lyrics(lyrics.get("lyrics_body") or "")
        if not text or not looks_like_lyrics(text, min_lines=2, min_words=8):
            return None
        return LyricsResult(provider=self.label, plain=text, artist=artist, title=title,
                            score=75.0, tier=2, partial=True)

    def identify(self, snippet: str) -> list[IdentifiedSong]:
        body = self._call("track.search", q_lyrics=snippet, page_size=5, f_has_lyrics=1,
                          s_track_rating="desc")
        out: list[IdentifiedSong] = []
        for item in (body.get("track_list") or []) if isinstance(body, dict) else []:
            track = (item or {}).get("track") or {}
            if track.get("track_name") and track.get("artist_name"):
                out.append(IdentifiedSong(track["artist_name"], track["track_name"], self.label))
        return out


PROVIDER_CLASSES: dict[str, type[LyricsProvider]] = {
    cls.id: cls for cls in (
        LrclibProvider, BetterLyricsProvider, RcLyricsBandProvider, MegalobizProvider,
        NetEaseProvider, QQMusicProvider, GeniusProvider, LyricsOvhProvider, MusixmatchProvider,
    )
}


# ---------------------------------------------------------------------- chain
@dataclass
class FetchOutcome:
    """Everything the chain learned about one track."""

    synced: Optional[LyricsResult] = None      # Tier 1 hit: ready to save
    plain: Optional[LyricsResult] = None       # best official text: needs alignment
    instrumental: bool = False
    errors: dict[str, str] = field(default_factory=dict)
    providers_tried: list[str] = field(default_factory=list)

    @property
    def kind(self) -> Optional[MatchKind]:
        if self.synced:
            return MatchKind.SYNCED
        if self.plain:
            return MatchKind.PLAIN_TEXT
        return MatchKind.INSTRUMENTAL if self.instrumental else None

    @property
    def all_providers_failed(self) -> bool:
        return bool(self.providers_tried) and len(self.errors) == len(self.providers_tried)

    @property
    def reference_text(self) -> str:
        """Official words for aligning: cleaned plain text, else the synced text stripped
        of its timestamps (a mashup component only needs the words)."""
        if self.plain and self.plain.plain:
            return self.plain.plain
        if self.synced and self.synced.synced_lrc:
            return strip_to_plain(self.synced.synced_lrc)
        return ""


class LyricsFetcher:
    """Runs Tier 1, then Tier 2, for one track (or one named component of a mashup)."""

    def __init__(self, settings: Settings, log: Optional[LogFn] = None,
                 cancel: Optional[threading.Event] = None) -> None:
        self.settings = settings
        self._log = log or (lambda tag, msg, level: None)
        self._cancel = cancel or threading.Event()
        self.http = HttpClient(timeout=settings.request_timeout)
        self.providers: list[LyricsProvider] = []
        skipped: list[LyricsProvider] = []
        for pid, cls in PROVIDER_CLASSES.items():          # canonical chain order
            if pid not in settings.enabled_providers:
                continue
            provider = cls(self.http, settings)
            (self.providers if provider.available() else skipped).append(provider)
        for provider in skipped:
            self._log(TAG_PLAIN, f"{provider.label} skipped (no API key configured)", "info")

    def close(self) -> None:
        self.http.close()

    # -------------------------------------------------------------- one tier
    def _run_tier(self, tier: int, query: TrackQuery, duration: Optional[float],
                  outcome: FetchOutcome) -> None:
        tag = TAG_T1 if tier == 1 else TAG_PLAIN
        artist, title = query.primary
        shown = f"{artist} - {title}" if artist else title
        for provider in (p for p in self.providers if p.tier == tier):
            if self._cancel.is_set():
                return
            if tier == 2 and outcome.plain and not outcome.plain.partial:
                return  # a complete official text is enough
            outcome.providers_tried.append(provider.label)
            self._log(tag, f"Searching {provider.label} for “{shown}”…", "info")
            try:
                hit = provider.search(query)
            except ProviderError as exc:
                outcome.errors[provider.label] = str(exc)
                self._log(tag, f"{provider.label} unavailable: {exc}", "warning")
                continue
            except Exception as exc:  # parser bug, odd payload... never abort the chain
                logger.exception("%s crashed", provider.label)
                outcome.errors[provider.label] = f"{exc.__class__.__name__}: {exc}"
                self._log(tag, f"{provider.label} failed: {exc.__class__.__name__}", "warning")
                continue

            if hit is None:
                self._log(tag, f"{provider.label}: no match", "info")
                continue
            if hit.kind is MatchKind.INSTRUMENTAL:
                outcome.instrumental = True
                self._log(tag, f"{provider.label}: marked instrumental", "info")
                return
            if hit.has_synced:
                ok, reason, _ = validate_synced(hit.synced_lrc or "", duration)
                if ok:
                    outcome.synced = hit
                    self._log("OK", f"{provider.label}: SYNCED lyrics found "
                              f"({hit.score:.0f}% match)", "info")
                    return
                self._log(tag, f"{provider.label}: rejected ({reason})", "warning")
            if hit.plain and looks_like_lyrics(hit.plain):
                better = (outcome.plain is None
                          or (outcome.plain.partial and not hit.partial)
                          or (hit.partial == outcome.plain.partial and hit.score > outcome.plain.score))
                if better:
                    outcome.plain = hit
                    note = " (excerpt only)" if hit.partial else ""
                    self._log(TAG_PLAIN, f"{provider.label}: PLAIN_TEXT lyrics found{note} "
                              f"({len(hit.plain.splitlines())} lines)", "info")

    # --------------------------------------------------------------- public
    def fetch_query(self, query: TrackQuery, duration: Optional[float] = None, *,
                    tier2: bool = True) -> FetchOutcome:
        outcome = FetchOutcome()
        if not query.variants:
            self._log(TAG_T1, "Nothing usable to search for (empty title)", "warning")
            return outcome
        self._run_tier(1, query, duration, outcome)
        if outcome.synced or outcome.instrumental or self._cancel.is_set():
            return outcome
        if tier2:
            self._run_tier(2, query, duration, outcome)
        return outcome

    def fetch(self, track: TrackInfo, *, tier2: bool = True) -> FetchOutcome:
        """Tier 1 (synced), then Tier 2 (plain text) for a normal track."""
        return self.fetch_query(TrackQuery.from_track(track), track.duration, tier2=tier2)

    def fetch_tier1(self, track: TrackInfo) -> FetchOutcome:
        return self.fetch(track, tier2=False)

    def fetch_tier2(self, track: TrackInfo, outcome: Optional[FetchOutcome] = None) -> FetchOutcome:
        outcome = outcome or FetchOutcome()
        self._run_tier(2, TrackQuery.from_track(track), track.duration, outcome)
        return outcome

    def fetch_for(self, artist: str, title: str) -> FetchOutcome:
        """Look up ONE named song (a mashup component). No duration: it is only part
        of the audio, so the track length says nothing about it."""
        return self.fetch_query(TrackQuery.from_names(artist, title), None)

    def identify_by_lyrics(self, snippet: str) -> list[IdentifiedSong]:
        """Which songs contain this sung phrase? (callers must still verify the text)"""
        found: list[IdentifiedSong] = []
        seen: set[tuple[str, str]] = set()
        for provider in self.providers:
            if self._cancel.is_set():
                break
            try:
                candidates = provider.identify(snippet)
            except ProviderError as exc:
                self._log(TAG_PLAIN, f"{provider.label} lyric search unavailable: {exc}", "warning")
                continue
            except Exception:
                logger.exception("%s identify crashed", provider.label)
                continue
            for song in candidates:
                key = (song.artist.casefold(), song.title.casefold())
                if key not in seen:
                    seen.add(key)
                    found.append(song)
        return found
