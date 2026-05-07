"""HTTP response caching for the spire-archive fetch step.

Wraps `requests` with `requests-cache` so re-fetching during development
doesn't re-hit the API. Cache lives at `~/.cache/sts-cards/api.sqlite` by
default and entries expire after 1 hour. Pass `use_cache=False` to bypass.

The retry layer (`tenacity`) sits *outside* the cache: cache hits are
served instantly and never trigger retries; only true network failures do.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 60 * 60  # 1 hour
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sts-cards"


def get_cache_path(cache_dir: Path | None = None) -> Path:
    """Resolve the cache database location.

    Honors `STS_CARDS_CACHE_DIR` env var; falls back to `~/.cache/sts-cards/`.
    """
    if cache_dir is not None:
        base = cache_dir
    elif "STS_CARDS_CACHE_DIR" in os.environ:
        base = Path(os.environ["STS_CARDS_CACHE_DIR"])
    else:
        base = DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / "api.sqlite"


def make_session(
    *,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> requests.Session:
    """Return a session with optional disk-backed caching.

    When `use_cache=False`, returns a plain `requests.Session` so the
    behavior is identical to `requests.get(...)` directly.
    """
    if not use_cache:
        log.info("HTTP cache: disabled")
        return requests.Session()

    try:
        from requests_cache import CachedSession
    except ImportError:
        log.warning(
            "requests-cache not installed; falling back to uncached session. "
            "Install with: pip install requests-cache"
        )
        return requests.Session()

    cache_path = get_cache_path(cache_dir)
    log.info("HTTP cache: %s (ttl=%ds)", cache_path, ttl_seconds)
    return CachedSession(
        cache_name=str(cache_path).removesuffix(".sqlite"),
        backend="sqlite",
        expire_after=ttl_seconds,
        # Only cache successful GETs
        allowable_methods=("GET",),
        allowable_codes=(200,),
        # Don't cache the auth header (none here, but defensive)
        ignored_parameters=["api_key"],
    )


def clear_cache(cache_dir: Path | None = None) -> None:
    """Wipe the on-disk cache. Useful before a known-needs-refresh run."""
    cache_path = get_cache_path(cache_dir)
    if cache_path.exists():
        cache_path.unlink()
        log.info("Cleared cache: %s", cache_path)
