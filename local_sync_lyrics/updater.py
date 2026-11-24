"""GitHub Releases update checker (pure Python, no Qt).

The GUI runs :func:`check_for_update` on a background ``QThread``; the CLI can use
it too.  Everything here is defensive: a missing network, a rate-limit, a renamed
repository or a malformed payload becomes an :class:`UpdateInfo` with ``error``
set, never an exception in the caller.

The repository is configured centrally in :mod:`local_sync_lyrics.config`
(``DEFAULT_UPDATE_REPO`` / the ``update_repo`` setting).
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import DEFAULT_UPDATE_REPO, GITHUB_RELEASES_URL, __version__

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com/repos/{repo}/releases/latest"
USER_AGENT = f"Lyricist/{__version__} (+{GITHUB_RELEASES_URL})"

# Callable injected by tests: ``opener(url, timeout) -> dict``.
Opener = Callable[[str, float], dict[str, Any]]


@dataclass
class ReleaseAsset:
    name: str = ""
    url: str = ""
    size: int = 0

    @property
    def suffix(self) -> str:
        name = self.name.lower()
        for suffix in (".tar.gz", ".appimage", ".dmg", ".pkg", ".exe", ".msi", ".zip", ".deb"):
            if name.endswith(suffix):
                return suffix
        return ""


@dataclass
class UpdateInfo:
    """Outcome of one update check."""

    current: str = __version__
    latest: str = ""
    available: bool = False
    tag: str = ""
    name: str = ""
    notes: str = ""
    html_url: str = ""
    asset: Optional[ReleaseAsset] = None
    published_at: str = ""
    error: str = ""

    @property
    def download_url(self) -> str:
        if self.asset is not None and self.asset.url:
            return self.asset.url
        return self.html_url or GITHUB_RELEASES_URL

    @property
    def asset_name(self) -> str:
        return self.asset.name if self.asset is not None else ""

    def summary(self) -> str:
        if self.error:
            return f"Could not check for updates: {self.error}"
        if not self.available:
            return f"You are up to date (v{self.current})."
        return f"Lyricist v{self.latest} is available (you have v{self.current})."


# --------------------------------------------------------------------- versions
def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(value or "").strip().lstrip("vV").replace("-", ".").split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def is_newer(current: str, latest: str) -> bool:
    """True when *latest* is a newer semantic version than *current*."""
    current, latest = (current or "").strip(), (latest or "").strip()
    if not latest:
        return False
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(latest.lstrip("vV")) > Version(current.lstrip("vV"))
        except InvalidVersion:
            pass
    except ImportError:  # packaging is part of the runtime requirements; stay safe anyway
        pass
    return _version_tuple(latest) > _version_tuple(current)


# ------------------------------------------------------------------- transport
def _default_opener(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        payload = response.read()
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("the release payload is not an object")
    return data


def api_url(repo: str = DEFAULT_UPDATE_REPO) -> str:
    repo = (repo or DEFAULT_UPDATE_REPO).strip().strip("/")
    return API_ROOT.format(repo=repo)


def fetch_latest_release(repo: str = DEFAULT_UPDATE_REPO, *, timeout: float = 8.0,
                         opener: Optional[Opener] = None) -> dict[str, Any]:
    """GET ``/releases/latest`` and return the decoded JSON object."""
    return (opener or _default_opener)(api_url(repo), timeout)


# ---------------------------------------------------------------------- assets
def _platform_key(platform: Optional[str] = None) -> str:
    value = (platform or sys.platform or "").lower()
    if value.startswith("win"):
        return "windows"
    if value.startswith("darwin") or value == "mac":
        return "macos"
    return "linux"


def select_asset(assets: Any, platform: Optional[str] = None) -> Optional[ReleaseAsset]:
    """Pick the most useful downloadable asset for this OS.

    ``assets`` is the GitHub ``assets`` array.  Source archives are ignored; a
    native installer wins over a zip, which wins over anything else.
    """
    if not isinstance(assets, list):
        return None
    preferred = {
        "windows": (".exe", ".msi", ".zip"),
        "macos": (".dmg", ".pkg", ".zip"),
        "linux": (".appimage", ".tar.gz", ".deb", ".zip"),
    }[_platform_key(platform)]
    candidates: list[ReleaseAsset] = []
    for raw in assets:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "") or "")
        url = str(raw.get("browser_download_url", "") or raw.get("url", "") or "")
        if not name or not url or name.lower().startswith(("source code", "source-code")):
            continue
        candidates.append(ReleaseAsset(name=name, url=url,
                                       size=int(raw.get("size", 0) or 0)))
    for suffix in preferred:
        for asset in candidates:
            if asset.name.lower().endswith(suffix):
                return asset
    return candidates[0] if candidates else None


def parse_release(payload: dict[str, Any], *, current: str = __version__,
                  platform: Optional[str] = None) -> UpdateInfo:
    """Turn a GitHub release object into an :class:`UpdateInfo`."""
    tag = str(payload.get("tag_name", "") or payload.get("name", "") or "").strip()
    latest = tag.lstrip("vV") or str(payload.get("name", "") or "").strip()
    asset = select_asset(payload.get("assets"), platform)
    info = UpdateInfo(
        current=current,
        latest=latest,
        tag=tag,
        name=str(payload.get("name", "") or tag),
        notes=str(payload.get("body", "") or ""),
        html_url=str(payload.get("html_url", "") or GITHUB_RELEASES_URL),
        asset=asset,
        published_at=str(payload.get("published_at", "") or ""),
    )
    info.available = bool(latest) and is_newer(current, latest)
    return info


# ------------------------------------------------------------------------ check
def check_for_update(current: str = __version__, repo: str = DEFAULT_UPDATE_REPO,
                     *, timeout: float = 8.0, opener: Optional[Opener] = None,
                     platform: Optional[str] = None) -> UpdateInfo:
    """Check the latest GitHub release. Never raises."""
    try:
        payload = fetch_latest_release(repo, timeout=timeout, opener=opener)
    except urllib.error.HTTPError as exc:
        reason = "no releases published yet" if exc.code == 404 else f"HTTP {exc.code}"
        logger.info("update check failed: %s", reason)
        return UpdateInfo(current=current, error=reason)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.info("update check failed: %s", exc)
        return UpdateInfo(current=current, error=str(getattr(exc, "reason", exc)))
    except (ValueError, TypeError) as exc:
        logger.info("update check returned malformed data: %s", exc)
        return UpdateInfo(current=current, error=f"malformed release data ({exc})")
    try:
        return parse_release(payload, current=current, platform=platform)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("release parsing failed", exc_info=True)
        return UpdateInfo(current=current, error=f"could not parse the release ({exc})")


def release_notes(info: UpdateInfo, *, limit: int = 6000) -> str:
    notes = (info.notes or "").strip()
    if not notes:
        return "No changelog was attached to this release."
    return notes if len(notes) <= limit else notes[:limit].rstrip() + "\n\n…"
