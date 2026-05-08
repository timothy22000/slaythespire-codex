"""Tests for extract_art.py.

No real game files are involved. We synthesize a tiny zip with the same
internal layout as desktop-1.0.jar, mock the GDRE Tools subprocess, and
assert the extraction + attach + diagnostic logic.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from PIL import Image

from sts_cards.extract_art import (
    JAR_PATHS,
    attach_art_to_cards,
    diagnose_jar,
    extract_jar_to_memory,
    extract_pck_to_memory,
    locate_jar,
    locate_pck,
    sample_dimensions,
)

# --- helpers -----------------------------------------------------------


def _png_bytes(
    size: tuple[int, int] = (4, 4),
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
) -> bytes:
    """Tiny in-memory PNG for fixture use — no disk I/O."""
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_jar(tmp_path: Path) -> Path:
    """A zip mirroring desktop-1.0.jar's real portrait layout: nested
    `<color>/<type>/<lowercase_stem>.png` paths under each resolution."""
    jar = tmp_path / "desktop-1.0.jar"
    with zipfile.ZipFile(jar, "w") as zf:
        zf.writestr("images/cards/red/attack/strike.png", _png_bytes((256, 256)))
        zf.writestr("images/cards/red/skill/defend.png", _png_bytes((256, 256)))
        zf.writestr("images/1024Portraits/red/attack/strike.png", _png_bytes((1024, 1024)))
        zf.writestr("images/1024Portraits/red/skill/defend.png", _png_bytes((1024, 1024)))
        zf.writestr("images/1024Portraits/red/attack/bash.png", _png_bytes((1024, 1024)))
        zf.writestr("META-INF/MANIFEST.MF", b"manifest")
        zf.writestr("images/icons/some_icon.png", _png_bytes((32, 32)))
    return jar


@pytest.fixture
def fake_cards_parquet(tmp_path: Path) -> Path:
    """Mirror the spire-archive ID convention: SCREAMING_SNAKE_CASE ids,
    Title-Case names. Defend_R / Strike_R share their JAR portrait with
    the other character variants via the trailing `_R` suffix strip."""
    out = tmp_path / "sts1_cards.parquet"
    pd.DataFrame([
        {"id": "STRIKE_R", "name": "Strike", "type": "Attack"},
        {"id": "DEFEND_R", "name": "Defend", "type": "Skill"},
        {"id": "BASH",     "name": "Bash",   "type": "Attack"},
        {"id": "SLIMED",   "name": "Slimed", "type": "Status"},  # no art
    ]).to_parquet(out, index=False)
    return out


# --- locate_jar / locate_pck -------------------------------------------


def test_locate_jar_with_explicit_path(fake_jar: Path):
    assert locate_jar(fake_jar) == fake_jar


def test_locate_jar_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        locate_jar(tmp_path / "nope.jar")


def test_locate_pck_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        locate_pck(tmp_path / "nope.pck")


# --- extract_jar_to_memory ---------------------------------------------


def test_extract_jar_high_resolution(fake_jar: Path):
    art = extract_jar_to_memory(fake_jar, resolution="high")
    assert set(art) == {"strike", "defend", "bash"}
    with Image.open(io.BytesIO(art["strike"])) as im:
        assert im.size == (1024, 1024)


def test_extract_jar_low_resolution(fake_jar: Path):
    art = extract_jar_to_memory(fake_jar, resolution="low")
    assert set(art) == {"strike", "defend"}
    with Image.open(io.BytesIO(art["strike"])) as im:
        assert im.size == (256, 256)


def test_extract_jar_rejects_unknown_resolution(fake_jar: Path):
    with pytest.raises(ValueError, match="resolution"):
        extract_jar_to_memory(fake_jar, resolution="medium")


def test_extract_jar_empty_when_no_cards(tmp_path: Path):
    empty = tmp_path / "empty.jar"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("not/a/card/path/file.png", _png_bytes())
    assert extract_jar_to_memory(empty, resolution="high") == {}


# --- diagnose_jar ------------------------------------------------------


def test_diagnose_jar_reports_match_rate(fake_jar: Path, fake_cards_parquet: Path):
    result = diagnose_jar(fake_jar, fake_cards_parquet)
    assert result.game == "sts1"
    assert result.extraction_source == "jar"
    assert result.n_cards_in_parquet == 4
    assert result.n_matched == 3                        # Slimed has no art
    assert result.match_rate == pytest.approx(0.75)
    assert result.most_common_path_prefix == JAR_PATHS["high"]
    assert "SLIMED" in result.unmatched_in_parquet


# --- attach_art_to_cards -----------------------------------------------


def _image_bytes(cell):
    """Pull the bytes out of an HF-style image struct cell.

    After attach_art_to_cards, the column shape is `struct<bytes, path>`,
    so each cell is either None or a dict with a `bytes` key.
    """
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell  # legacy path


def test_attach_art_adds_columns_in_place(fake_cards_parquet: Path):
    """The attach step joins parquet ids (SCREAMING_SNAKE_CASE) to JAR
    stems (lowercase) via `candidate_keys`: STRIKE_R/DEFEND_R strip the
    _R suffix, BASH lowercases. SLIMED has no art."""
    art = {"strike": _png_bytes(), "defend": _png_bytes(), "bash": _png_bytes()}
    stats = attach_art_to_cards(fake_cards_parquet, art, resolution="high")

    assert stats["n_total"] == 4
    assert stats["n_matched"] == 3

    df = pd.read_parquet(fake_cards_parquet)
    assert "image" in df.columns
    assert "image_resolution" in df.columns
    assert {"id", "name", "type"}.issubset(df.columns)
    slimed = df.loc[df["id"] == "SLIMED"].iloc[0]
    assert _image_bytes(slimed["image"]) is None
    assert slimed["image_resolution"] is None
    strike = df.loc[df["id"] == "STRIKE_R"].iloc[0]
    assert strike["image_resolution"] == "high"
    # Image column is the HF-compatible struct, not raw bytes.
    assert isinstance(strike["image"], dict)
    assert "bytes" in strike["image"] and "path" in strike["image"]


def test_attach_art_is_idempotent(fake_cards_parquet: Path):
    art = {"strike": _png_bytes(), "defend": _png_bytes(), "bash": _png_bytes()}
    attach_art_to_cards(fake_cards_parquet, art, resolution="high")
    attach_art_to_cards(fake_cards_parquet, art, resolution="high")

    df = pd.read_parquet(fake_cards_parquet)
    assert len(df) == 4
    assert sum(c == "image" for c in df.columns) == 1


def test_attach_art_bytes_round_trip(fake_cards_parquet: Path):
    """Bytes written to Parquet round-trip bit-identically and decode to PIL."""
    raw = _png_bytes((8, 8), (0, 255, 0, 255))
    attach_art_to_cards(fake_cards_parquet, {"strike": raw}, resolution="high")
    df = pd.read_parquet(fake_cards_parquet)
    out_bytes = _image_bytes(df.loc[df["id"] == "STRIKE_R", "image"].iloc[0])
    assert out_bytes == raw
    with Image.open(io.BytesIO(out_bytes)) as im:
        assert im.size == (8, 8)


def test_attach_art_writes_hf_image_struct_schema(fake_cards_parquet: Path):
    """Parquet schema for `image` must be struct<bytes, path> and the
    Arrow-level `huggingface` metadata must mark it as an Image feature.
    Both are needed for the HF dataset viewer to render thumbnails."""
    import json
    import pyarrow.parquet as pq

    art = {"strike": _png_bytes()}
    attach_art_to_cards(fake_cards_parquet, art, resolution="high")

    parquet = pq.ParquetFile(fake_cards_parquet)
    schema = parquet.schema_arrow
    image_field = schema.field("image")
    assert str(image_field.type) == "struct<bytes: binary, path: string>"

    meta = schema.metadata or {}
    assert b"huggingface" in meta, "missing HF feature metadata in parquet schema"
    info = json.loads(meta[b"huggingface"].decode("utf-8"))
    assert info["info"]["features"]["image"] == {"_type": "Image"}


# --- extract_pck_to_memory (subprocess mocked) -------------------------


def test_extract_pck_uses_cached_extraction(tmp_path: Path):
    """If the GDRE work_dir already has the sentinel, we skip the subprocess
    and just walk the directory."""
    work = tmp_path / "gdre_out"
    work.mkdir()
    (work / "card_art").mkdir()
    (work / "card_art" / "Strike_R.png").write_bytes(_png_bytes())
    (work / "card_art" / "Defend_R.png").write_bytes(_png_bytes())
    (work / "card_art" / "monster_slime.png").write_bytes(_png_bytes())  # ignored
    (work / ".gdre_complete").touch()

    fake_pck = tmp_path / "sts2.pck"
    fake_pck.write_bytes(b"fake-pck")

    with patch("sts_cards.extract_art._resolve_gdre", return_value="gdre_tools"), \
         patch("sts_cards.extract_art._run_gdre") as run_mock:
        art = extract_pck_to_memory(
            fake_pck,
            work_dir=work,
            card_ids=["Strike_R", "Defend_R", "Bash"],
        )
    # Subprocess should NOT have been invoked because the sentinel exists
    run_mock.assert_not_called()
    # Only known card ids are picked up; the monster PNG is filtered out
    assert set(art) == {"Strike_R", "Defend_R"}


def test_extract_pck_runs_gdre_when_no_sentinel(tmp_path: Path):
    work = tmp_path / "gdre_out"
    fake_pck = tmp_path / "sts2.pck"
    fake_pck.write_bytes(b"fake-pck")

    def fake_run(_binary, _pck, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "Strike_R.png").write_bytes(_png_bytes())

    with patch("sts_cards.extract_art._resolve_gdre", return_value="gdre_tools"), \
         patch("sts_cards.extract_art._run_gdre", side_effect=fake_run) as run_mock:
        art = extract_pck_to_memory(fake_pck, work_dir=work, card_ids=None)

    run_mock.assert_called_once()
    assert "Strike_R" in art
    # Sentinel is dropped after a fresh run
    assert (work / ".gdre_complete").exists()


# --- sample_dimensions -------------------------------------------------


def test_sample_dimensions_returns_size():
    art = {"a": _png_bytes((32, 64))}
    assert sample_dimensions(art) == (32, 64)


def test_sample_dimensions_empty_returns_none():
    assert sample_dimensions({}) is None


# --- HF datasets round-trip (optional — only when `datasets` installed) -


def test_image_column_decodes_via_datasets(fake_cards_parquet: Path):
    """When `datasets` is available, the bytes column round-trips into PIL
    via the `Image()` feature. Skipped when the library isn't installed
    (CI environment minimum)."""
    pytest.importorskip("datasets")
    from datasets import Dataset, Features, Value
    from datasets import Image as HFImage

    art = {"strike": _png_bytes((16, 16))}
    attach_art_to_cards(fake_cards_parquet, art, resolution="high")
    df = pd.read_parquet(fake_cards_parquet).dropna(subset=["image"])

    ds = Dataset.from_pandas(
        df[["id", "image", "image_resolution"]],
        features=Features({
            "id": Value("string"),
            "image": HFImage(),
            "image_resolution": Value("string"),
        }),
        preserve_index=False,
    )
    row = ds[0]
    # `Image()` feature decodes bytes to a PIL Image automatically
    assert hasattr(row["image"], "size")
    assert row["image"].size == (16, 16)
