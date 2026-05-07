"""Tests for the cache module.

We don't hit any real URL — just verify the session factory does what
we expect. requests-cache is exercised more heavily in its own tests.
"""

import os
from pathlib import Path

import pytest
import requests

from sts_cards.cache import (
    DEFAULT_TTL_SECONDS,
    clear_cache,
    get_cache_path,
    make_session,
)


def test_get_cache_path_uses_default(tmp_path, monkeypatch):
    monkeypatch.delenv("STS_CARDS_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        "sts_cards.cache.DEFAULT_CACHE_DIR", tmp_path / "default-cache"
    )
    p = get_cache_path()
    assert p.parent == tmp_path / "default-cache"
    assert p.name == "api.sqlite"
    assert p.parent.exists()  # directory created


def test_get_cache_path_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STS_CARDS_CACHE_DIR", str(tmp_path / "envcache"))
    p = get_cache_path()
    assert p.parent == tmp_path / "envcache"


def test_get_cache_path_honors_explicit_arg(tmp_path):
    p = get_cache_path(tmp_path / "explicit")
    assert p.parent == tmp_path / "explicit"


def test_make_session_no_cache_returns_plain_session(tmp_path):
    s = make_session(use_cache=False, cache_dir=tmp_path)
    assert isinstance(s, requests.Session)
    # Plain session has no `.cache` attribute
    assert not hasattr(s, "cache")


def test_make_session_with_cache_returns_cached(tmp_path):
    pytest.importorskip("requests_cache")
    s = make_session(use_cache=True, cache_dir=tmp_path, ttl_seconds=300)
    # CachedSession exposes a `.cache` attribute
    assert hasattr(s, "cache")


def test_clear_cache_removes_db(tmp_path):
    pytest.importorskip("requests_cache")
    # Create a session to populate the file
    make_session(use_cache=True, cache_dir=tmp_path)
    cache_path = get_cache_path(tmp_path)
    # The db file is created lazily by requests-cache on first request,
    # so create an empty file to assert removal works.
    cache_path.write_text("")
    assert cache_path.exists()
    clear_cache(tmp_path)
    assert not cache_path.exists()


def test_clear_cache_is_safe_when_missing(tmp_path):
    # Should not raise even if the cache file doesn't exist
    clear_cache(tmp_path / "nonexistent")
