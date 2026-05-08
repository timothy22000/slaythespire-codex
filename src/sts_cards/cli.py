"""Typer CLI for the sts-cards pipeline.

Each command operates on ONE game at a time, since the two games are
shipped as separate HuggingFace datasets.

Examples:
    sts-cards fetch sts1
    sts-cards fetch sts2 --sts-game-version v0.103.0 --no-cache
    sts-cards embed sts1
    sts-cards visualize sts2
    sts-cards search sts1 --card "Strike"
    sts-cards search sts2 --query "deal damage and apply vulnerable"
    sts-cards croissant sts1 --repo myname/slay-the-spire-1-cards
    sts-cards upload sts1 --repo myname/slay-the-spire-1-cards
    sts-cards cache-clear
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from . import DEFAULT_MODEL, DEFAULT_TASK_INSTRUCTION, GAMES, __version__

app = typer.Typer(
    name="sts-cards",
    help="Slay the Spire 1 + 2 card dataset and embedding pipeline.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_game(game: str) -> None:
    if game not in GAMES:
        raise typer.BadParameter(f"game must be one of {GAMES}")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def fetch(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
    lang: str = typer.Option("en", "--lang"),
    sts_game_version: str | None = typer.Option(
        None, "--sts-game-version",
        help="STS2 Steam version e.g. 'v0.103.0' — recorded in provenance.",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache",
        help="Bypass the HTTP cache and re-fetch from the API.",
    ),
) -> None:
    """Fetch one game's cards from spire-archive.com."""
    from .fetch import fetch_game
    _check_game(game)
    out_path = fetch_game(
        game, out_dir=out_dir, lang=lang,
        sts_game_version=sts_game_version,
        use_cache=not no_cache,
    )
    typer.echo(f"Wrote {out_path}")


@app.command()
def embed(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    task_instruction: str = typer.Option(
        DEFAULT_TASK_INSTRUCTION, "--task-instruction",
        help="Pass '' to disable.",
    ),
    matryoshka_dim: int | None = typer.Option(None, "--matryoshka-dim"),
    batch_size: int = typer.Option(16, "--batch-size"),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    """Embed one game's cards. Reads {game}_cards.parquet, writes
    {game}_cards_with_embeddings.parquet plus updated provenance."""
    from .embed import embed_game
    _check_game(game)
    in_path = out_dir / f"{game}_cards.parquet"
    if not in_path.exists():
        raise typer.BadParameter(
            f"{in_path} not found — run `sts-cards fetch {game}` first"
        )
    out_path = embed_game(
        in_path, out_dir,
        game=game, model_id=model, task_instruction=task_instruction,
        matryoshka_dim=matryoshka_dim, batch_size=batch_size, device=device,
    )
    typer.echo(f"Wrote {out_path}")


@app.command()
def visualize(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
    n_neighbors: int = typer.Option(15, "--n-neighbors"),
    min_dist: float = typer.Option(0.1, "--min-dist"),
    seed: int = typer.Option(42, "--seed"),
    no_html: bool = typer.Option(False, "--no-html"),
) -> None:
    """UMAP-project one game's embeddings to 2D and write an HTML plot."""
    from .visualize import project_2d
    _check_game(game)
    in_path = out_dir / f"{game}_cards_with_embeddings.parquet"
    if not in_path.exists():
        raise typer.BadParameter(
            f"{in_path} not found — run `sts-cards embed {game}` first"
        )
    out_html = None if no_html else (out_dir / f"{game}_umap_2d.html")
    project_2d(in_path, out_html=out_html,
               n_neighbors=n_neighbors, min_dist=min_dist, seed=seed)
    typer.echo(f"Updated {in_path}")
    if out_html:
        typer.echo(f"Plot → {out_html}")


@app.command()
def search(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
    card: str | None = typer.Option(None, "--card"),
    query: str | None = typer.Option(None, "--query"),
    k: int = typer.Option(10, "--k"),
) -> None:
    """Find similar cards within one game's index."""
    from .search import load_index, search_by_card, search_by_text
    if not (card or query):
        raise typer.BadParameter("Provide --card NAME or --query TEXT")
    _check_game(game)

    in_path = out_dir / f"{game}_cards_with_embeddings.parquet"
    df, emb, prov = load_index(in_path)
    typer.echo(f"Index: {len(df)} {game} cards × {emb.shape[1]}D")
    if prov and prov.embed:
        typer.echo(f"  model: {prov.embed.model_id}")

    if card:
        results = search_by_card(df, emb, card, k=k)
    else:
        assert query is not None
        results = search_by_text(df, emb, query, prov, k=k)

    typer.echo(f"\nTop {k}:")
    with pd.option_context("display.max_colwidth", 80, "display.width", 160):
        typer.echo(results.to_string(index=False))


@app.command()
def croissant(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    kind: str = typer.Option(
        "cards", "--kind",
        help="Which dataset to describe: 'cards' or 'embeddings'.",
    ),
    repo: str = typer.Option(
        ..., "--repo",
        help="HF repo id (used in the Croissant `url` field), "
             "e.g. user/slay-the-spire-1-cards or user/slay-the-spire-1-card-embeddings",
    ),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
) -> None:
    """Generate a Croissant JSON-LD descriptor for one (game, kind) dataset."""
    from .croissant import write_croissant
    _check_game(game)
    if kind not in ("cards", "embeddings"):
        raise typer.BadParameter("kind must be 'cards' or 'embeddings'")

    parquet_name = (
        f"{game}_cards.parquet" if kind == "cards"
        else f"{game}_embeddings.parquet"
    )
    parquet_path = out_dir / parquet_name
    if not parquet_path.exists():
        raise typer.BadParameter(f"{parquet_path} not found")

    out_path = write_croissant(parquet_path, game=game, kind=kind, repo_id=repo)
    typer.echo(f"Wrote {out_path}")


@app.command()
def upload(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    kind: str = typer.Option(
        "cards", "--kind",
        help="Which repo to upload to: 'cards' or 'embeddings'.",
    ),
    repo: str = typer.Option(..., "--repo",
                             help="HF repo id, e.g. user/slay-the-spire-1-cards"),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
    readme: Path = typer.Option(
        None, "--readme",
        help="Path to dataset card README.md "
             "(default: dataset_cards/{game}_{kind}_README.md)",
    ),
    private: bool = typer.Option(False, "--private"),
) -> None:
    """Upload one (game, kind) dataset to its HuggingFace repo."""
    from .upload import upload as upload_fn
    _check_game(game)
    if kind not in ("cards", "embeddings"):
        raise typer.BadParameter("kind must be 'cards' or 'embeddings'")

    if readme is None:
        readme = Path("dataset_cards") / f"{game}_{kind}_README.md"
    if not readme.exists():
        typer.echo(f"WARNING: dataset card not found at {readme}", err=True)
        readme = None  # type: ignore[assignment]

    upload_fn(repo=repo, game=game, kind=kind,
              out_dir=out_dir, readme=readme, private=private)


@app.command(name="cache-clear")
def cache_clear() -> None:
    """Wipe the on-disk HTTP cache used by `fetch`."""
    from .cache import clear_cache
    clear_cache()
    typer.echo("Cleared.")


@app.command(name="diagnose-art")
def diagnose_art(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
    jar_path: Path | None = typer.Option(None, "--jar-path",
                                         help="STS1: path to desktop-1.0.jar"),
    pck_path: Path | None = typer.Option(None, "--pck-path",
                                         help="STS2: path to sts2.pck"),
    gdre_tools_path: str = typer.Option("gdre_tools", "--gdre-tools-path",
                                        help="STS2: GDRE Tools binary"),
) -> None:
    """List candidate card-portrait paths in the local game files and
    report join rate against {game}_cards.parquet["id"].

    Run this before `extract-art` to verify the path layout. Output is
    written to {out_dir}/{game}_art_diagnostic.json.
    """
    import json as _json
    from .extract_art import diagnose_jar, diagnose_pck

    _check_game(game)
    cards_parquet = out_dir / f"{game}_cards.parquet"
    if not cards_parquet.exists():
        raise typer.BadParameter(
            f"{cards_parquet} not found — run `sts-cards fetch {game}` first"
        )

    if game == "sts1":
        from .extract_art import locate_jar
        jar = locate_jar(jar_path)
        result = diagnose_jar(jar, cards_parquet)
    else:
        from .extract_art import locate_pck
        pck = locate_pck(pck_path)
        result = diagnose_pck(pck, cards_parquet, gdre_tools_path=gdre_tools_path)

    out_path = out_dir / f"{game}_art_diagnostic.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(result.to_dict(), indent=2))
    typer.echo(_json.dumps(result.to_dict(), indent=2))
    typer.echo(f"\nWrote {out_path}")
    typer.echo(
        f"\nMatch rate: {result.n_matched}/{result.n_cards_in_parquet} "
        f"({100 * result.match_rate:.1f}%)"
    )
    if result.match_rate < 0.95:
        typer.echo(
            "\nWARNING: match rate is below 95%. Inspect "
            "`unmatched_in_source_sample` and `unmatched_in_parquet_sample` "
            "before running `extract-art`.",
            err=True,
        )


@app.command(name="extract-art")
def extract_art(
    game: str = typer.Argument(..., help=f"One of {GAMES}"),
    out_dir: Path = typer.Option(Path("output"), "--out-dir"),
    jar_path: Path | None = typer.Option(None, "--jar-path",
                                         help="STS1: path to desktop-1.0.jar"),
    pck_path: Path | None = typer.Option(None, "--pck-path",
                                         help="STS2: path to sts2.pck"),
    gdre_tools_path: str = typer.Option("gdre_tools", "--gdre-tools-path",
                                        help="STS2: GDRE Tools binary"),
    resolution: str = typer.Option("high", "--resolution",
                                   help="'high' or 'low' (STS1 only — STS2 ships one)"),
) -> None:
    """Extract card portraits from local game files and attach them as
    an `image` column on {game}_cards.parquet."""
    from .extract_art import (
        attach_art_to_cards, extract_jar_to_memory, extract_pck_to_memory,
        locate_jar, locate_pck, sample_dimensions, sha256_file,
        _gdre_version,
    )
    from .provenance import ArtProvenance, DatasetProvenance, now_iso

    _check_game(game)
    if resolution not in ("high", "low"):
        raise typer.BadParameter("resolution must be 'high' or 'low'")
    cards_parquet = out_dir / f"{game}_cards.parquet"
    if not cards_parquet.exists():
        raise typer.BadParameter(
            f"{cards_parquet} not found — run `sts-cards fetch {game}` first"
        )

    if game == "sts1":
        source_path = locate_jar(jar_path)
        art = extract_jar_to_memory(source_path, resolution=resolution)
        extraction_source = "jar"
        gdre_version = None
    else:
        source_path = locate_pck(pck_path)
        # Prefilter to known card ids so cache + attach skip non-card PNGs.
        ids = set(pd.read_parquet(cards_parquet)["id"].astype(str))
        art = extract_pck_to_memory(
            source_path, resolution=resolution,
            gdre_tools_path=gdre_tools_path, card_ids=ids,
        )
        extraction_source = "pck"
        try:
            gdre_version = _gdre_version(gdre_tools_path)
        except Exception:
            gdre_version = None

    if not art:
        raise typer.BadParameter(
            f"No card art extracted from {source_path}. "
            f"Run `sts-cards diagnose-art {game}` to check the path layout."
        )

    stats = attach_art_to_cards(cards_parquet, art, resolution=resolution)
    typer.echo(
        f"Attached art to {cards_parquet.name}: "
        f"{stats['n_matched']}/{stats['n_total']} matched "
        f"({100 * stats['match_rate']:.1f}%)"
    )

    # Update provenance
    prov_path = out_dir / f"{game}_provenance.json"
    if prov_path.exists():
        prov = DatasetProvenance.read(prov_path)
    else:
        typer.echo(f"WARNING: no provenance at {prov_path} — skipping art provenance",
                   err=True)
        return

    prov.art = ArtProvenance(
        extraction_source=extraction_source,
        source_file_sha256=sha256_file(source_path),
        extracted_at=now_iso(),
        n_art_files=int(stats["n_matched"]),
        n_cards_total=int(stats["n_total"]),
        resolution=resolution,
        image_dimensions=sample_dimensions(art),
        gdre_tools_version=gdre_version,
    )
    prov.write(prov_path)
    typer.echo(f"Updated provenance → {prov_path}")


if __name__ == "__main__":
    app()
