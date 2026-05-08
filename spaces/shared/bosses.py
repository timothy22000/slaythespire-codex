"""Boss data for STS1 and STS2 + LLM-generated strategy reasoning.

Bosses are hardcoded per game per act. STS1 is canonical and stable;
STS2 is Early Access and shifts across patches — STS2 entries are
labeled best-effort and may drift. Each entry captures:
  - name
  - act
  - threat: 1-2 sentences on the boss's key threat pattern
  - counters: 1-2 sentences on what kinds of cards/strategies handle it

Strategy reasoning calls a chat model via HF Inference Providers using
the same HF_TOKEN secret the Synergy Inspector uses for vision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


REASONING_MODEL = "Qwen/Qwen2.5-72B-Instruct"


@dataclass
class Boss:
    name: str
    act: int
    threat: str
    counters: str


# STS1 bosses are stable across releases (game launched 2019, content frozen).
# Each act has 3 boss variants except Act 4 (just Corrupt Heart).
STS1_BOSSES: list[Boss] = [
    # Act 1
    Boss("Slime Boss", 1,
         "Splits into two large slimes at half HP and again into smaller slimes; can stun-lock with Bash for 35 damage.",
         "Single-target burst before split, or AoE to manage adds. Vulnerable + heavy attack on the split turn is the classic answer."),
    Boss("Hexaghost", 1,
         "Charges over 6 turns then unleashes 6 attacks scaling with current HP (deals roughly half max HP). Inflicts Burn statuses.",
         "Status-card removal helps; otherwise stack block on turn 6 or burst it down before charge completes."),
    Boss("Lagavulin", 1,
         "Sleeps for 3 turns, then wakes and applies Weak + Frail every other turn. High HP, hits hard once awake.",
         "Burst him down before he wakes, or stack permanent buffs while he sleeps. Artifact handles the debuffs."),
    # Act 2
    Boss("Bronze Automaton", 2,
         "Summons two Bronze Orb minions (each draws 2 cards from your discard for itself), then casts Hyper Beam (45 damage) on a 2-turn cycle.",
         "Kill the orbs fast or the boss turns your own discard against you. Block the Hyper Beam turn or burst before second cast."),
    Boss("The Champ", 2,
         "Phase 2 at 50% HP makes him aggressive: Heavy Slash, Faceslam, Anger; cleanses debuffs. Hits very hard.",
         "Front-load Vulnerable before he cleanses; in phase 2 prioritize block and burst windows."),
    Boss("The Collector", 2,
         "Summons two Torch Heads each turn (deal 7-9 damage each); only takes damage when Torch Heads are present.",
         "AoE damage to clear torches and chip the boss. Powers and scaling skills that survive multiple turns shine."),
    # Act 3
    Boss("Awakened One", 3,
         "Phase 1 dies to ~300 damage; revives at full HP with +20 Strength and Curiosity (gains Strength when you play Powers). Long fight.",
         "Avoid stacking Powers. Bring durable scaling that doesn't rely on Powers to ramp. Strength-Down or Weaken trivializes phase 2."),
    Boss("Donu and Deca", 3,
         "Two enemies. Donu gives both +3 Strength every turn; Deca shuffles 2 Dazed into your draw every turn. ~250 HP each.",
         "Kill Donu first to stop the Strength snowball. Or AoE both. Card-removal events help against the Dazed flood."),
    Boss("Time Eater", 3,
         "After your 12th card played in a turn, ends your turn. Heals to 50% at half HP. Massive single-target hits.",
         "Big swings over many small ones. Strength-scaling plays, single high-damage cards, Powers that don't count toward the limit."),
    # Act 4
    Boss("Corrupt Heart", 4,
         "Final boss with Beat of Death (every card played → 1 damage), Invincible (caps damage taken per turn), Time Warp echo phase, and 4 Debuff per turn.",
         "Need both block-stacking and a clean kill turn that breaks Invincible. Artifact stack against the debuffs. Long-game decks usually lose."),
]

# STS2 boss roster — best effort as of the indexed snapshot. STS2 is in Early
# Access; bosses are added/rebalanced with patches. Treat threats and counters
# as approximate. Verify against in-game info if exact matchup is critical.
STS2_BOSSES: list[Boss] = [
    Boss("Slime Boss", 1,
         "Returning from STS1 with similar split mechanics, retuned for STS2's faster pace.",
         "Same answer as STS1: burst before split or AoE adds."),
    Boss("Hexaghost", 1,
         "Returning from STS1; charge-attack pattern preserved.",
         "Status removal or burn-it-down before charge completes."),
    Boss("Guardian", 1,
         "STS2 act-1 boss with a defensive mode that reduces incoming damage; switches to offensive bursts.",
         "Time burst windows around mode switches. Vulnerable helps."),
    Boss("Bronze Automaton", 2,
         "Returning, same orb-summoning + Hyper Beam pattern.",
         "Kill orbs fast; block the Hyper Beam turn."),
    Boss("Collector", 2,
         "Returning. Torch Heads + only-damageable-with-adds-present pattern.",
         "AoE the torches, chip the boss."),
    Boss("Soul Reaper", 2,
         "STS2 boss: drains soul tokens from the player; uses a soul economy that interacts with Necrobinder mechanics.",
         "Disrupt soul accumulation; Necrobinder benefits, others must outpace the drain."),
    Boss("Awakened One", 3,
         "Returning with the two-phase pattern. STS2 scaling tweaks.",
         "Avoid Power spam in phase 2; Strength-Down ideal."),
    Boss("Time Eater", 3,
         "Returning: ends turn at 12th card played.",
         "Few big plays per turn; Strength-scaling shines."),
    Boss("The Reclaimer", 3,
         "STS2 boss that retrieves your discarded cards and uses them. Punishes Exhaust-heavy decks indirectly by depleting your own pool.",
         "Self-discard or Ethereal cards become liabilities. Decks that don't churn the discard pile fare better."),
    Boss("Corrupt Heart", 4,
         "Final boss returning from STS1; mechanics retained.",
         "Block stack + clean kill turn; Artifact for debuffs."),
]


BOSSES = {"sts1": STS1_BOSSES, "sts2": STS2_BOSSES}


def bosses_for_prompt(game: str) -> str:
    """Return a markdown-formatted boss list suitable for embedding in an LLM prompt."""
    bosses = BOSSES.get(game, [])
    by_act: dict[int, list[Boss]] = {}
    for b in bosses:
        by_act.setdefault(b.act, []).append(b)

    lines: list[str] = []
    for act in sorted(by_act):
        lines.append(f"\nAct {act}:")
        for b in by_act[act]:
            lines.append(f"- **{b.name}** — {b.threat} *Counter:* {b.counters}")
    return "\n".join(lines).strip()


REASONING_PROMPT_TEMPLATE = """\
You are an expert {game_label} player analyzing a deck for the {character_label}.

The user described their playstyle goal as:
"{user_prompt}"

Here is the deck the algorithm built (cards listed with cost, type, and description):

{deck_listing}

Composition: {composition_summary}

Boss landscape for {game_label}:
{boss_landscape}

Write a concise strategic analysis in three sections, using markdown headings:

### Strategy summary
2-3 sentences explaining what this deck wants to do each turn.

### Key synergies
3-5 short bullets naming the strongest interactions between cards in this deck.

### Boss matchups
For each act, write 1-2 sentences specifically about how this deck handles that act's bosses given the cards present. Be honest about weaknesses — call out specific bosses the deck struggles against and explain why.

Be direct and grounded in the actual cards. Don't invent cards that aren't in the deck. If the deck doesn't actually fit the user's stated playstyle, say so in the Strategy summary.
"""


def _format_deck_listing(picks: list[Any], max_cards: int = 30) -> str:
    """Compact deck listing for the LLM prompt. Deduplicate locked starters."""
    from collections import Counter
    name_counts: Counter[tuple[str, str, str, str]] = Counter()
    by_key: dict[tuple[str, str, str, str], Any] = {}
    for p in picks:
        key = (p.name, p.type_, p.cost, p.description)
        name_counts[key] += 1
        by_key.setdefault(key, p)

    lines: list[str] = []
    for (name, type_, cost, desc), count in name_counts.items():
        cost_label = "X" if cost == "-1" else cost
        copies = f" ×{count}" if count > 1 else ""
        desc_short = (desc or "").strip()
        if len(desc_short) > 140:
            desc_short = desc_short[:137] + "..."
        lines.append(f"- {name}{copies} ({type_}, cost {cost_label}): {desc_short}")
        if len(lines) >= max_cards:
            lines.append(f"... and {len(picks) - sum(name_counts.values())} more")
            break
    return "\n".join(lines)


def _format_composition(result: Any) -> str:
    types = result.type_count
    a, s, p = types.get("Attack", 0), types.get("Skill", 0), types.get("Power", 0)
    curve = result.curve_count
    return (
        f"{a} Attacks, {s} Skills, {p} Powers; "
        f"curve {curve.get('low', 0)} low / {curve.get('mid', 0)} mid / {curve.get('high', 0)} high; "
        f"avg cost {result.avg_cost:.1f}; avg similarity {result.avg_sim_picks:.2f} on prompt-driven picks."
    )


def generate_strategy(
    game: str,
    character: str,
    user_prompt: str,
    result: Any,  # DeckResult — typed loosely to avoid circular import
) -> str | None:
    """Call HF Inference Providers to produce strategic reasoning for the deck.

    Returns Markdown text, or None if HF_TOKEN is not configured. Raises on
    actual API failures so the caller can render an error.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        return None

    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="auto", api_key=hf_token)

    game_label = "Slay the Spire 1" if game == "sts1" else "Slay the Spire 2"
    character_label = character.title()

    prompt = REASONING_PROMPT_TEMPLATE.format(
        game_label=game_label,
        character_label=character_label,
        user_prompt=(user_prompt or "(no specific playstyle described)"),
        deck_listing=_format_deck_listing(result.picks),
        composition_summary=_format_composition(result),
        boss_landscape=bosses_for_prompt(game),
    )

    resp = client.chat.completions.create(
        model=REASONING_MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()
