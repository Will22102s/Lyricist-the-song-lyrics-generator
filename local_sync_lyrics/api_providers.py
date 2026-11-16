"""Sequential multi-provider synced-lyrics search.

Chain (default): **LRCLIB -> BetterLyrics -> Megalobiz**.

Design rules
------------
* Queries are sanitised first (``text_utils.query_variants``): YouTube IDs,
  "(Official Video)", track numbers, "Artist - " prefixes ... never reach a server.
* Every candidate is *scored* (fuzzy artist/title + duration) and *validated*
  (does it look like real, track-length LRC?) before it is accepted.
* A failing provider (offline, 5xx, rate limit, layout change) is logged and
  skipped - it can never abort the chain or the batch.
* Network calls have timeouts, bounded retries and per-provider throttling.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .lrc_utils import build_lrc, parse_lrc, parse_ttml, validate_synced
from .models import LyricsResult, TrackInfo
from .text_utils import query_variants, similarity

logger = logging.getLogger(__name__)

USER_AGENT = "LocalSyncLyrics/2.0 (https://github.com/local-sync-lyrics; desktop lyric sync tool)"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
LogFn = Callable[[str, str, str], None]  # (tag, message, level)


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
        return cls(
            variants=query_variants(track.artist, track.title),
            album=track.album,
            duration=track.duration,
        )

    @property
    def primary(self) -> tuple[str, str]:
        return self.variants[0] if self.variants else ("", "")


def score_candidate(
    query: TrackQuery,
    cand_artist: str,
    cand_title: str,
    cand_duration: Optional[float],
) -> float:
    """0-100 confidence that a candidate is the track we are looking for."""
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
            allowed_methods=("GET",), respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, */*"})

    def get(self, url: str, params: Optional[dict] = None,
            headers: Optional[dict] = None) -> Optional[requests.Response]:
        """GET -> Response, or ``None`` for 'no such thing' (404/400/204).

        Raises :class:`ProviderError` for transport problems / rate limits / 5xx.
        """
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"network error: {exc.__class__.__name__}") from exc
        if response.status_code in (404, 400, 204, 410):
            return None
        if response.status_code in (401, 403):
            raise ProviderError(f"access denied (HTTP {response.status_code})")
        if response.status_code == 429:
            raise ProviderError("rate limited (HTTP 429)")
        if response.status_code >= 400:
            raise ProviderError(f"HTTP {response.status_code}")
        return response

    def get_json(self, url: str, params: Optional[dict] = None):
        response = self.get(url, params)
        if response is None:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError("invalid JSON in response") from exc

    def close(self) -> None:
        self.session.close()


# ------------------------------------------------------------------ providers
class LyricsProvider(ABC):
    id = "base"
    label = "Base"
    min_interval = 0.25  # seconds between requests (politeness)

    def __init__(self, http: HttpClient, settings: Settings) -> None:
        self.http = http
        self.settings = settings
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @abstractmethod
    def search(self, query: TrackQuery) -> Optional[LyricsResult]:
        """Return the best hit or ``None``. May raise :class:`ProviderError`."""


class LrclibProvider(LyricsProvider):
    """https://lrclib.net - open database of synced lyrics (best first choice)."""

    id, label = "lrclib", "LRCLIB"
    BASE = "https://lrclib.net/api"

    def _to_result(self, item: dict, query: TrackQuery) -> Optional[LyricsResult]:
        if not isinstance(item, dict):
            return None
        artist, title = item.get("artistName") or "", item.get("trackName") or ""
        duration = item.get("duration")
        score = score_candidate(
            query, artist, title, float(duration) if isinstance(duration, (int, float)) else None
        )
        return LyricsResult(
            provider=self.label,
            synced_lrc=(item.get("syncedLyrics") or None),
            plain=(item.get("plainLyrics") or None),
            instrumental=bool(item.get("instrumental")),
            artist=artist,
            title=title,
            score=score,
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
            # The API does not echo the matched title, so confidence comes from
            # the duration check above; report a neutral-high score.
            return LyricsResult(provider=self.label, synced_lrc=lrc, artist=artist,
                                title=title, score=80.0)
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
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError("beautifulsoup4 is not installed") from exc

        artist, title = query.primary
        term = f"{artist} {title}".strip()
        if not term:
            return None
        self._throttle()
        response = self.http.get(
            f"{self.BASE}/search/all",
            {"qry": term, "searchButton.x": 0, "searchButton.y": 0},
            headers=self._headers(),
        )
        if response is None:
            return None
        soup = BeautifulSoup(response.text, "html.parser")

        ranked: list[tuple[float, str, str]] = []
        for anchor in soup.find_all("a", class_="entity_name"):
            href = anchor.get("href") or ""
            if not href.startswith("/lrc/maker/"):
                continue
            label = anchor.get_text(" ", strip=True)
            score = similarity(term, label)
            ranked.append((score, href, label))
        ranked.sort(key=lambda r: -r[0])

        for score, href, label in ranked[:3]:
            if score < self.settings.min_match_score - 10:
                break
            self._throttle()
            page = self.http.get(f"{self.BASE}{href}", headers=self._headers())
            if page is None:
                continue
            details = BeautifulSoup(page.text, "html.parser").find("div", class_="lyrics_details")
            if details is None:
                continue
            lines = parse_lrc(details.get_text("\n"))
            if not lines:
                continue
            lrc = build_lrc(lines, artist=artist, title=title, album=query.album,
                            duration=query.duration, source=self.label)
            ok, reason, _ = validate_synced(lrc, query.duration)
            if not ok:
                logger.debug("Megalobiz '%s' rejected: %s", label, reason)
                continue
            return LyricsResult(provider=self.label, synced_lrc=lrc, artist=artist,
                                title=title, score=float(score))
        return None


PROVIDER_CLASSES: dict[str, type[LyricsProvider]] = {
    LrclibProvider.id: LrclibProvider,
    BetterLyricsProvider.id: BetterLyricsProvider,
    MegalobizProvider.id: MegalobizProvider,
}


# ---------------------------------------------------------------------- chain
@dataclass
class FetchOutcome:
    synced: Optional[LyricsResult] = None
    plain: Optional[LyricsResult] = None
    instrumental: bool = False
    errors: dict[str, str] = field(default_factory=dict)
    providers_tried: list[str] = field(default_factory=list)

    @property
    def all_providers_failed(self) -> bool:
        return bool(self.providers_tried) and len(self.errors) == len(self.providers_tried)


class LyricsFetcher:
    """Runs the provider chain for one track."""

    def __init__(self, settings: Settings, log: Optional[LogFn] = None,
                 cancel: Optional[threading.Event] = None) -> None:
        self.settings = settings
        self._log = log or (lambda tag, msg, level: None)
        self._cancel = cancel or threading.Event()
        self.http = HttpClient(timeout=settings.request_timeout)
        self.providers: list[LyricsProvider] = [
            PROVIDER_CLASSES[pid](self.http, settings)
            for pid in settings.enabled_providers
            if pid in PROVIDER_CLASSES
        ]

    def close(self) -> None:
        self.http.close()

    def fetch(self, track: TrackInfo) -> FetchOutcome:
        outcome = FetchOutcome()
        query = TrackQuery.from_track(track)
        if not query.variants:
            self._log("API", "Nothing usable to search for (empty title)", "warning")
            return outcome

        artist, title = query.primary
        shown = f"{artist} - {title}" if artist else title
        for provider in self.providers:
            if self._cancel.is_set():
                break
            outcome.providers_tried.append(provider.label)
            self._log("API", f"Searching {provider.label} for “{shown}”…", "info")
            try:
                hit = provider.search(query)
            except ProviderError as exc:
                outcome.errors[provider.label] = str(exc)
                self._log("API", f"{provider.label} unavailable: {exc}", "warning")
                continue
            except Exception as exc:  # parser bug, odd payload... never abort the chain
                logger.exception("%s crashed", provider.label)
                outcome.errors[provider.label] = f"{exc.__class__.__name__}: {exc}"
                self._log("API", f"{provider.label} failed: {exc.__class__.__name__}", "warning")
                continue

            if hit is None:
                self._log("API", f"{provider.label}: no match", "info")
                continue
            if hit.instrumental and not hit.has_synced:
                outcome.instrumental = True
                self._log("API", f"{provider.label}: marked instrumental", "info")
                return outcome
            if hit.has_synced:
                ok, reason, _ = validate_synced(hit.synced_lrc or "", track.duration)
                if ok:
                    outcome.synced = hit
                    self._log("OK", f"{provider.label}: synced lyrics found "
                              f"({hit.score:.0f}% match)", "info")
                    return outcome
                self._log("API", f"{provider.label}: rejected ({reason})", "warning")
            if hit.plain and (outcome.plain is None or hit.score > outcome.plain.score):
                outcome.plain = hit
                self._log("API", f"{provider.label}: only unsynced lyrics", "info")
        return outcome


def lrc_to_json_debug(lrc: str) -> str:  # tiny helper for tests / debugging
    return json.dumps([(l.time, l.text) for l in parse_lrc(lrc)], ensure_ascii=False)
