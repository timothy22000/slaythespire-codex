"""Extract card portrait art from local STS1 / STS2 game files.

Game files are NOT redistributed. The user must own a Steam install of
each game; extraction runs on their machine and produces raw PNG bytes
that get attached as an `image` column on the existing cards Parquet.
HuggingFace's `datasets.Image()` feature decodes those bytes back to PIL
on `load_dataset()`.

  - STS1 → desktop-1.0.jar (a zip — Python stdlib `zipfile` is enough)
  - STS2 → sts2.pck (Godot package — needs GDRE Tools subprocess)

`diagnose_*` helpers list what's inside each file and report the
join-rate against `cards.parquet["id"]` so we can verify the schema
before committing to extraction.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path layouts inside the game files
# ---------------------------------------------------------------------------

# STS1 desktop-1.0.jar lays card art out under two directories.
# `images/cards/` is the in-game render path (~256×256). `images/1024Portraits/`
# is the high-res variant used when the card detail panel is open.
JAR_PATHS = {
    "low": "images/cards/",
    "high": "images/1024Portraits/",
}

# STS2 portraits live at res://images/packed/card_portraits/<character>/<card>.png
# inside the PCK (Godot import descriptors). The actual texture bytes are stored
# as .ctex (compressed texture) at res://.godot/imported/<name>-<hash>.ctex —
# `gdre_tools --recover` converts them back to PNG and writes them to the
# logical source path.
#
# Recovery needs BOTH the descriptors (which give it the source paths) AND the
# imported .ctex blobs (which carry the bytes). Globs match files actually in
# the PCK, not source paths — that's why "res://**/card_portraits/**/*.png"
# alone returns zero matches: source PNGs aren't in the PCK, only the
# .png.import descriptors and the .ctex blobs are.
PCK_CARD_PORTRAIT_DIR = "images/packed/card_portraits"
PCK_INCLUDE_GLOBS = (
    "res://**/card_portraits/**/*.png.import",
    "res://.godot/imported/*.ctex",
)
PCK_CARD_PATH_HINTS = (PCK_CARD_PORTRAIT_DIR + "/",)


# Parquet card ids (from spire-archive) and JAR portrait stems use different
# naming conventions: SCREAMING_SNAKE_CASE vs lowercase, with occasional
# word-split disagreements. We try multiple normalizations per row, falling
# back to a small explicit alias table for known one-off mismatches.
STS1_ID_ALIASES: dict[str, str] = {
    # parquet id  -> JAR stem
    "DROPKICK":      "drop_kick",
    "FORCE_FIELD":   "forcefield",
    "HYPERBEAM":     "hyper_beam",
    "J_A_X":         "jax",
    "LESSONLEARNED": "lessons_learned",
    "LOCKON":        "lock_on",
    "MULTI_CAST":    "multicast",
    "THUNDERCLAP":   "thunder_clap",
    "WREATHOFFLAME": "wreathe_of_flame",  # plain typo in the JAR asset name
    # STS2 — Necrobinder card that morphs into one of three variants.
    # Pick the attack flavor as the canonical portrait; the other two
    # (mad_science_power, mad_science_skill) stay unmapped on purpose.
    "MAD_SCIENCE":   "mad_science_attack",
}


def _normalize_name(s: str) -> str:
    """Card display name → JAR-stem candidate.

    "Battle Hymn"   → "battle_hymn"
    "Multi-Cast"    → "multi_cast"
    "Ascender's Bane" → "ascenders_bane"
    """
    return re.sub(r"[^a-z0-9_]+", "_", s.lower().replace("'", "")).strip("_")


def _strip_character_suffix(card_id: str) -> str:
    """Remove the trailing _R/_G/_B/_P character variant marker — STS1
    has e.g. STRIKE_R/G/B/P all sharing one `strike.png`."""
    return re.sub(r"_[RGBPW]$", "", card_id.lower())


def candidate_keys(card_id: str, name: str | None = None) -> list[str]:
    """Yield JAR-stem candidates for a parquet (id, name) row in priority
    order: explicit alias > stripped id > normalized name > raw lowercase id."""
    cands: list[str] = []
    if card_id in STS1_ID_ALIASES:
        cands.append(STS1_ID_ALIASES[card_id])
    cands.append(_strip_character_suffix(card_id))
    if name:
        cands.append(_normalize_name(name))
    cands.append(card_id.lower())
    # de-dupe preserving order
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


@dataclass
class DiagnosticResult:
    """Output of `diagnose_jar` / `diagnose_pck`. Returned to the caller
    and (in the CLI) also serialized to JSON so we can compare runs."""
    game: str
    extraction_source: str  # "jar" or "pck"
    candidate_paths: list[str]
    n_images_in_source: dict[str, int]  # path-prefix → count
    n_cards_in_parquet: int
    n_matched: int
    match_rate: float
    matched_ids_sample: list[str]
    unmatched_in_source: list[str]
    unmatched_in_parquet: list[str]
    most_common_path_prefix: str | None = None

    def to_dict(self) -> dict:
        return {
            "game": self.game,
            "extraction_source": self.extraction_source,
            "candidate_paths": self.candidate_paths,
            "n_images_in_source": self.n_images_in_source,
            "n_cards_in_parquet": self.n_cards_in_parquet,
            "n_matched": self.n_matched,
            "match_rate": round(self.match_rate, 4),
            "matched_ids_sample": self.matched_ids_sample[:20],
            "unmatched_in_source_sample": self.unmatched_in_source[:50],
            "unmatched_in_parquet_sample": self.unmatched_in_parquet[:50],
            "most_common_path_prefix": self.most_common_path_prefix,
        }


# ---------------------------------------------------------------------------
# Locating the game files
# ---------------------------------------------------------------------------


def _steam_default_jar_paths() -> list[Path]:
    """Per-platform best-guess Steam install paths for desktop-1.0.jar."""
    home = Path.home()
    if sys.platform == "darwin":
        return [
            home / "Library/Application Support/Steam/steamapps/common"
                 / "SlayTheSpire/SlayTheSpire.app/Contents/Resources/desktop-1.0.jar",
        ]
    if sys.platform.startswith("win"):
        return [
            Path(r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire\desktop-1.0.jar"),
            Path(r"C:\Program Files\Steam\steamapps\common\SlayTheSpire\desktop-1.0.jar"),
        ]
    # Linux / others
    return [
        home / ".steam/steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
        home / ".local/share/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
    ]


def _steam_default_pck_paths() -> list[Path]:
    """Per-platform best-guess Steam install paths for the STS2 .pck.

    The Steam folder is "Slay the Spire 2" (with spaces) and the package
    file is "Slay the Spire 2.pck" inside the .app bundle on macOS.
    Older notes referring to "sts2.pck" / "SlayTheSpire2/" are kept as
    fallbacks for non-Steam layouts.
    """
    home = Path.home()
    if sys.platform == "darwin":
        return [
            home / "Library/Application Support/Steam/steamapps/common"
                 / "Slay the Spire 2/SlayTheSpire2.app/Contents/Resources/Slay the Spire 2.pck",
            home / "Library/Application Support/Steam/steamapps/common"
                 / "SlayTheSpire2/SlayTheSpire2.app/Contents/Resources/sts2.pck",
        ]
    if sys.platform.startswith("win"):
        return [
            Path(r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\Slay the Spire 2.pck"),
            Path(r"C:\Program Files\Steam\steamapps\common\Slay the Spire 2\Slay the Spire 2.pck"),
            Path(r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire2\sts2.pck"),
        ]
    return [
        home / ".steam/steam/steamapps/common/Slay the Spire 2/Slay the Spire 2.pck",
        home / ".local/share/Steam/steamapps/common/Slay the Spire 2/Slay the Spire 2.pck",
        home / ".steam/steam/steamapps/common/SlayTheSpire2/sts2.pck",
    ]


def locate_jar(jar_path: Path | None = None) -> Path:
    """Resolve the STS1 desktop-1.0.jar path.

    Explicit `jar_path` wins. Otherwise probe known Steam locations.
    Raises FileNotFoundError with the candidates we tried.
    """
    if jar_path is not None:
        p = Path(jar_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"STS1 JAR not found at {p}")
        return p
    candidates = _steam_default_jar_paths()
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find desktop-1.0.jar in any default Steam location. "
        "Pass --jar-path explicitly. Tried:\n  " + tried
    )


def locate_pck(pck_path: Path | None = None) -> Path:
    """Resolve the STS2 sts2.pck path."""
    if pck_path is not None:
        p = Path(pck_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"STS2 PCK not found at {p}")
        return p
    candidates = _steam_default_pck_paths()
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find sts2.pck in any default Steam location. "
        "Pass --pck-path explicitly. Tried:\n  " + tried
    )


# ---------------------------------------------------------------------------
# JAR extraction (STS1)
# ---------------------------------------------------------------------------


def _read_jar_at_prefix(jar_path: Path, prefix: str) -> dict[str, bytes]:
    """Return {card_id: png_bytes} for every PNG directly under `prefix`."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(jar_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if not name.startswith(prefix) or not name.lower().endswith(".png"):
                continue
            # No subdirectories — the layout is flat under the prefix.
            stem = Path(name).stem
            if not stem:
                continue
            out[stem] = zf.read(info)
    return out


def extract_jar_to_memory(
    jar_path: Path, resolution: str = "high"
) -> dict[str, bytes]:
    """Pull card portraits from desktop-1.0.jar into memory.

    Returns a `{card_id: png_bytes}` mapping. Bytes are kept raw (never
    decoded) — they go straight into the Parquet `image` column where
    HuggingFace's `Image()` feature handles PIL decoding for consumers.
    """
    if resolution not in JAR_PATHS:
        raise ValueError(
            f"resolution must be one of {sorted(JAR_PATHS)}, got {resolution!r}"
        )
    prefix = JAR_PATHS[resolution]
    log.info("Reading PNGs from %s under prefix %r", jar_path.name, prefix)
    art = _read_jar_at_prefix(jar_path, prefix)
    log.info("Loaded %d portrait PNGs from JAR", len(art))
    return art


def diagnose_jar(jar_path: Path, cards_parquet: Path) -> DiagnosticResult:
    """List every PNG under the candidate paths in the JAR and report the
    join rate against `cards.parquet["id"]` using `candidate_keys()`."""
    cards_df = pd.read_parquet(cards_parquet)
    rows = list(cards_df[["id", "name"]].itertuples(index=False, name=None))

    # Bucket JAR stems by their containing path prefix so we can report
    # which directory carries the cards.
    stems_per_prefix: dict[str, set[str]] = {p: set() for p in JAR_PATHS.values()}
    with zipfile.ZipFile(jar_path, "r") as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".png"):
                continue
            for prefix in JAR_PATHS.values():
                if info.filename.startswith(prefix):
                    stems_per_prefix[prefix].add(Path(info.filename).stem)

    n_per_path = {p: len(s) for p, s in stems_per_prefix.items()}

    def _matched_per_prefix(stems: set[str]) -> tuple[set[str], list[str]]:
        ok, unmatched = set(), []
        for cid, name in rows:
            for c in candidate_keys(str(cid), str(name) if name else None):
                if c in stems:
                    ok.add(str(cid))
                    break
            else:
                unmatched.append(str(cid))
        return ok, unmatched

    matched_per_prefix = {p: _matched_per_prefix(s) for p, s in stems_per_prefix.items()}
    primary_prefix = max(matched_per_prefix, key=lambda p: len(matched_per_prefix[p][0]))
    matched, unmatched_in_parquet = matched_per_prefix[primary_prefix]

    # Stems present in the JAR's primary directory but never resolved by any
    # parquet row's candidate keys.
    used_keys: set[str] = set()
    for cid, name in rows:
        for c in candidate_keys(str(cid), str(name) if name else None):
            if c in stems_per_prefix[primary_prefix]:
                used_keys.add(c)
                break
    unmatched_in_source = sorted(stems_per_prefix[primary_prefix] - used_keys)

    return DiagnosticResult(
        game="sts1",
        extraction_source="jar",
        candidate_paths=list(JAR_PATHS.values()),
        n_images_in_source=n_per_path,
        n_cards_in_parquet=len(rows),
        n_matched=len(matched),
        match_rate=len(matched) / max(len(rows), 1),
        matched_ids_sample=sorted(matched)[:20],
        unmatched_in_source=unmatched_in_source,
        unmatched_in_parquet=sorted(unmatched_in_parquet),
        most_common_path_prefix=primary_prefix,
    )


# ---------------------------------------------------------------------------
# PCK extraction (STS2) — shells out to GDRE Tools
# ---------------------------------------------------------------------------


_GDRE_INSTALL_HINT = (
    "GDRE Tools (https://github.com/bruvzg/gdsdecomp) is required for STS2 "
    "art extraction. Install it and pass --gdre-tools-path "
    "(or put `gdre_tools` on your PATH)."
)


def _resolve_gdre(gdre_tools_path: str = "gdre_tools") -> str:
    """Return an absolute path to the GDRE binary, or raise."""
    resolved = shutil.which(gdre_tools_path)
    if resolved is None:
        raise FileNotFoundError(
            f"GDRE Tools binary {gdre_tools_path!r} not found. " + _GDRE_INSTALL_HINT
        )
    return resolved


def _gdre_version(binary: str) -> str | None:
    try:
        r = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=10,
        )
        out = (r.stdout + r.stderr).strip().splitlines()
        return out[0] if out else None
    except (subprocess.SubprocessError, OSError):
        return None


def _gdre_cache_dir(pck_path: Path) -> Path:
    """Cache directory keyed by the PCK file's content hash so re-extracts
    of an unchanged PCK skip the slow subprocess."""
    sha = _sha256(pck_path)
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return base / "sts-cards" / "gdre" / sha[:16]


def _run_gdre(
    binary: str,
    pck_path: Path,
    out_dir: Path,
    include_globs: tuple[str, ...] = PCK_INCLUDE_GLOBS,
) -> None:
    """Run GDRE Tools to recover card-portrait PNGs from the PCK.

    Uses `--recover` (not `--extract`) because Godot stores assets as
    `.ctex` compressed textures; recovery is what reconverts them back
    to source PNGs at their logical paths. The `--include` globs scope
    work to portraits — without them, recovery on a 1.7 GB PCK takes
    tens of minutes and produces gigabytes of unrelated assets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary, "--headless",
        f"--recover={pck_path}",
        f"--output={out_dir}",
    ]
    cmd.extend(f"--include={glob}" for glob in include_globs)
    log.info("Running GDRE Tools: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"gdre_tools failed (exit {r.returncode}). stderr: {r.stderr.strip()}"
        )


def _walk_pngs_into_id_map(root: Path, card_ids: set[str] | None) -> dict[str, bytes]:
    """Walk the GDRE output directory and pull PNGs whose stem is a
    known card id (when `card_ids` is provided) or every PNG (when None).

    Prefers walking the canonical card_portraits/ subtree if it exists
    (faster — skips the rest of the recovered project) and falls back
    to a full rglob so synthetic test fixtures still work.
    """
    portraits_root = root / PCK_CARD_PORTRAIT_DIR
    walk_root = portraits_root if portraits_root.exists() else root
    out: dict[str, bytes] = {}
    for path in walk_root.rglob("*.png"):
        stem = path.stem
        if card_ids is not None and stem not in card_ids:
            continue
        out[stem] = path.read_bytes()
    return out


def extract_pck_to_memory(
    pck_path: Path,
    resolution: str = "high",
    gdre_tools_path: str = "gdre_tools",
    work_dir: Path | None = None,
    card_ids: Iterable[str] | None = None,
) -> dict[str, bytes]:
    """Run GDRE Tools to expand sts2.pck and load card portrait PNGs into memory.

    `resolution` is accepted for API symmetry with `extract_jar_to_memory`,
    but Godot's portrait pipeline ships one resolution per asset path —
    we simply record the choice in provenance.

    GDRE extraction is slow (minutes); results are cached on disk keyed
    by `sha256(pck)` under `~/.cache/sts-cards/gdre/`.
    """
    if resolution not in {"high", "low"}:
        raise ValueError(f"resolution must be 'high' or 'low', got {resolution!r}")
    binary = _resolve_gdre(gdre_tools_path)

    cache_dir = _gdre_cache_dir(pck_path) if work_dir is None else Path(work_dir)
    sentinel = cache_dir / ".gdre_complete"
    if not sentinel.exists():
        _run_gdre(binary, pck_path, cache_dir)
        sentinel.touch()
    else:
        log.info("Using cached GDRE extraction at %s", cache_dir)

    ids = set(card_ids) if card_ids is not None else None
    art = _walk_pngs_into_id_map(cache_dir, ids)
    log.info("Loaded %d portrait PNGs from PCK extraction", len(art))
    return art


def diagnose_pck(
    pck_path: Path,
    cards_parquet: Path,
    gdre_tools_path: str = "gdre_tools",
    work_dir: Path | None = None,
) -> DiagnosticResult:
    """Run GDRE Tools, walk the recovered `card_portraits/` tree, and
    report the join rate against `cards.parquet["id"]` using the same
    multi-candidate keying as the JAR path (lower(id), suffix-stripped,
    name-normalized, alias-table).
    """
    cards_df = pd.read_parquet(cards_parquet)
    rows = list(cards_df[["id", "name"]].itertuples(index=False, name=None))

    binary = _resolve_gdre(gdre_tools_path)
    cache_dir = _gdre_cache_dir(pck_path) if work_dir is None else Path(work_dir)
    sentinel = cache_dir / ".gdre_complete"
    if not sentinel.exists():
        _run_gdre(binary, pck_path, cache_dir)
        sentinel.touch()

    portraits_root = cache_dir / PCK_CARD_PORTRAIT_DIR
    walk_root = portraits_root if portraits_root.exists() else cache_dir

    # Collect every PNG stem under the portrait tree, plus per-character bucket
    # counts for the diagnostic report.
    all_stems: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    for path in walk_root.rglob("*.png"):
        all_stems.add(path.stem)
        rel = path.relative_to(walk_root)
        # Per-character bucket = first directory component (e.g. "regent")
        bucket = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        bucket_counts[bucket] += 1

    matched_ids: set[str] = set()
    used_keys: set[str] = set()
    unmatched_parquet: list[str] = []
    for cid, name in rows:
        cid_s, name_s = str(cid), str(name) if name else None
        for c in candidate_keys(cid_s, name_s):
            if c in all_stems:
                matched_ids.add(cid_s)
                used_keys.add(c)
                break
        else:
            unmatched_parquet.append(cid_s)
    unmatched_source = sorted(all_stems - used_keys)

    return DiagnosticResult(
        game="sts2",
        extraction_source="pck",
        candidate_paths=list(PCK_CARD_PATH_HINTS),
        n_images_in_source={k: int(v) for k, v in bucket_counts.most_common(15)},
        n_cards_in_parquet=len(rows),
        n_matched=len(matched_ids),
        match_rate=len(matched_ids) / max(len(rows), 1),
        matched_ids_sample=sorted(matched_ids)[:20],
        unmatched_in_source=unmatched_source,
        unmatched_in_parquet=sorted(unmatched_parquet),
        most_common_path_prefix=PCK_CARD_PORTRAIT_DIR if portraits_root.exists() else None,
    )


# ---------------------------------------------------------------------------
# Attaching art bytes to the cards Parquet
# ---------------------------------------------------------------------------


def _png_dimensions(png_bytes: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(png_bytes)) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def attach_art_to_cards(
    cards_parquet: Path,
    art_bytes: dict[str, bytes],
    resolution: str = "high",
) -> dict[str, int | float]:
    """Add `image` (raw PNG bytes) and `image_resolution` columns to
    `cards.parquet`. Each row's id is joined via `candidate_keys(id, name)`
    so SCREAMING_SNAKE_CASE parquet ids resolve to lowercase JAR stems
    even when word boundaries differ. Cards without art get `image=None`.

    Idempotent: existing `image` / `image_resolution` columns are
    overwritten in place.
    """
    df = pd.read_parquet(cards_parquet)
    n_total = len(df)
    if "id" not in df.columns:
        raise ValueError(f"{cards_parquet} has no `id` column to join on")

    has_name = "name" in df.columns

    def _lookup(row: pd.Series) -> bytes | None:
        name = row["name"] if has_name else None
        for c in candidate_keys(str(row["id"]), str(name) if name else None):
            if c in art_bytes:
                return art_bytes[c]
        return None

    df["image"] = df.apply(_lookup, axis=1)
    df["image_resolution"] = df["image"].map(lambda b: resolution if b is not None else None)

    # Move image / image_resolution to the front (right after `id`) so the
    # HF dataset viewer surfaces the portrait thumbnail early in the row.
    front = [c for c in ("id", "image", "image_resolution") if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    n_matched = int(df["image"].notna().sum())
    df.to_parquet(cards_parquet, index=False)

    log.info(
        "Attached art to %s: %d/%d cards matched (%.1f%%)",
        cards_parquet.name, n_matched, n_total, 100.0 * n_matched / max(n_total, 1),
    )
    return {
        "n_total": n_total,
        "n_matched": n_matched,
        "match_rate": n_matched / max(n_total, 1),
    }


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """Public wrapper — used by the CLI to record provenance."""
    return _sha256(path)


def sample_dimensions(art_bytes: dict[str, bytes]) -> tuple[int, int] | None:
    """Pick a deterministic sample image and return its (width, height)."""
    if not art_bytes:
        return None
    first_id = sorted(art_bytes)[0]
    return _png_dimensions(art_bytes[first_id])
