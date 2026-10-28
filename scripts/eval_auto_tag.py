# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Measure auto-tag suggestion quality against the tags already in a library.

The library's own tags are used as ground truth, so no separate annotated
dataset is needed. Every entry is scored under four selection rules — facet rule
and hubness correction each on or off — so the effects can be compared directly
on the same embeddings.

Usage::

    python scripts/eval_auto_tag.py /path/to/library
    python scripts/eval_auto_tag.py /path/to/library --mode holdout --limit 500
    python scripts/eval_auto_tag.py /path/to/library --model google/embeddinggemma-2

Modes:

``cold``
    Entry text is built without any of its tags, mimicking an untagged image.
    Reports suggestion precision, recall and sibling conflicts.

``holdout``
    One applied tag per entry is hidden and treated as the answer; the remaining
    tags stay in the entry text, matching the shipped pipeline. Reports
    recall@k and the mean reciprocal rank of the hidden tag.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tagstudio.core.library.alchemy.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    TASK_TAGGING,
    EmbeddingConfig,
    build_entry_input,
    frame_cache_dir,
    select_tag_suggestions,
    tag_facets,
    tag_hubness,
    tag_label,
)
from tagstudio.core.library.alchemy.library import Library
from tagstudio.core.library.alchemy.models import Entry, Tag


@dataclass
class Tally:
    """Accumulated results for a single configuration."""

    entries: int = 0
    applied_tags: int = 0
    suggestions: int = 0
    hits: int = 0
    conflicts: int = 0
    reciprocal_rank: float = 0.0
    ranks: list[int] = field(default_factory=list)

    def add(
        self,
        *,
        selected: list[tuple[Tag, float]],
        applied: set[int],
        claimed: set[int],
        facets: dict[int, frozenset[int]],
        rank: int | None = None,
    ) -> None:
        self.entries += 1
        self.applied_tags += len(applied)
        self.suggestions += len(selected)

        used = set(claimed)
        for tag, _score in selected:
            tag_id = int(tag.id) if tag.id is not None else -1
            if tag_id in applied:
                self.hits += 1
            facets_of_tag = facets.get(tag_id, frozenset())
            if facets_of_tag & used:
                self.conflicts += 1
            used |= facets_of_tag

        if rank is not None:
            self.ranks.append(rank)
            self.reciprocal_rank += 1.0 / rank

    @property
    def precision(self) -> float:
        return self.hits / self.suggestions if self.suggestions else 0.0

    @property
    def recall(self) -> float:
        return self.hits / self.applied_tags if self.applied_tags else 0.0

    @property
    def mean_suggestions(self) -> float:
        return self.suggestions / self.entries if self.entries else 0.0

    @property
    def conflict_rate(self) -> float:
        return self.conflicts / self.entries if self.entries else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank / self.entries if self.entries else 0.0

    def recall_at(self, k: int) -> float:
        if not self.ranks:
            return 0.0
        return sum(1 for rank in self.ranks if rank <= k) / len(self.ranks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("library", type=Path, help="Path to a TagStudio library directory")
    parser.add_argument(
        "--mode",
        choices=("cold", "holdout"),
        default="cold",
        help="cold: score entries without their tags. holdout: hide one tag (default: cold)",
    )
    parser.add_argument("--limit", type=int, default=200, help="Max entries to score (0 = all)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Suggestion threshold")
    parser.add_argument("--max-per-entry", type=int, default=5, help="Max suggestions per entry")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10], help="Recall cutoffs")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-images", action="store_true", help="Ignore image content")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for the holdout choice")
    parser.add_argument(
        "--no-hubness",
        action="store_true",
        help="Only report the uncorrected rules (skips the cross-entry sample)",
    )
    return parser.parse_args(argv)


def entry_vector(
    index,
    entry: Entry,
    library: Library,
    exclude_tag_ids: set[int],
    use_images: bool,
) -> np.ndarray:
    model_input, _hash = build_entry_input(
        entry,
        TASK_TAGGING,
        library.library_dir,
        use_images,
        exclude_tag_ids,
        cache_dir=frame_cache_dir(library.library_dir),
    )
    return index.backend.encode([model_input])[0]


# (column label, one_per_facet, use_hubness)
CONFIGS: tuple[tuple[str, bool, bool], ...] = (
    ("facet+hub", True, True),
    ("facet", True, False),
    ("hub", False, True),
    ("plain", False, False),
)


def evaluate(
    index,
    library: Library,
    entries: list[Entry],
    tags: list[Tag],
    tag_vectors: np.ndarray,
    facets: dict[int, frozenset[int]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[dict[str, Tally], bool]:
    configs = [config for config in CONFIGS if not (args.no_hubness and config[2])]
    tallies = {config[0]: Tally() for config in configs}
    use_images = not args.no_images
    tag_ids = [int(tag.id) if tag.id is not None else -1 for tag in tags]
    candidate_ids = set(tag_ids)

    # Embed every entry once; the selection rules are then replayed over the same
    # vectors, so any difference in the report comes from the rules themselves.
    prepared: list[tuple[set[int], int | None, np.ndarray]] = []
    for done, entry in enumerate(entries, start=1):
        applied = {int(tag.id) for tag in entry.tags if tag.id is not None}
        applied_candidates = sorted(applied & candidate_ids)
        if not applied_candidates:
            continue

        held_out: int | None = None
        if args.mode == "holdout":
            held_out = rng.choice(applied_candidates)
            exclude = {held_out}
        else:
            exclude = set(applied)

        vector = entry_vector(index, entry, library, exclude, use_images)
        prepared.append((applied, held_out, vector))

        if done % 25 == 0:
            print(f"  embedded {done}/{len(entries)} entries", file=sys.stderr, flush=True)

    sample = np.vstack([vector for _applied, _held, vector in prepared]) if prepared else None
    hubness = tag_hubness(tag_vectors, sample) if sample is not None else None

    for applied, held_out, vector in prepared:
        raw = np.asarray(tag_vectors @ vector, dtype=np.float32)

        for name, one_per_facet, use_hubness in configs:
            tally = tallies[name]
            adjusted = raw - hubness if use_hubness and hubness is not None else raw
            min_margin: float | None = 0.0 if use_hubness and hubness is not None else None

            if args.mode == "holdout" and held_out is not None:
                visible = applied - {held_out}
                claimed = set()
                for tag_id in visible:
                    claimed |= facets.get(tag_id, frozenset())
            else:
                visible = set()
                claimed = set()

            # Rank of the hidden tag among everything this rule could still suggest.
            rank: int | None = None
            if held_out is not None:
                position = 0
                for raw_index in np.argsort(-adjusted):
                    candidate = tag_ids[int(raw_index)]
                    if candidate in applied and candidate != held_out:
                        continue
                    if float(adjusted[int(raw_index)]) < 0.0 and min_margin is not None:
                        break
                    position += 1
                    if candidate == held_out:
                        rank = position
                        break

            selected = select_tag_suggestions(
                adjusted,
                tags,
                threshold=args.threshold,
                max_per_entry=args.max_per_entry,
                existing_tag_ids=visible,
                facets=facets,
                one_per_facet=one_per_facet,
                raw_scores=raw,
                min_margin=min_margin,
            )
            tally.add(
                selected=selected,
                applied=set(applied) & candidate_ids,
                claimed=claimed,
                facets=facets,
                rank=rank,
            )

    return tallies, hubness is not None


def print_report(
    args: argparse.Namespace, tallies: dict[str, Tally], hubness_available: bool
) -> None:
    entries = next(iter(tallies.values())).entries
    print()
    print(
        f"Auto-tag evaluation | mode={args.mode} | entries={entries} | "
        f"threshold={args.threshold:.2f} | max/entry={args.max_per_entry}"
    )
    images = "off" if args.no_images else "on"
    print(f"model={args.model} | dimensions={args.dimensions} | images={images}")
    if not hubness_available:
        print("Hubness correction unavailable: too few entries to build a sample.")
    print()

    columns = list(tallies)
    print(f"{'metric':<32}" + "".join(f"{name:>13}" for name in columns))

    def row(label: str, values: list[str]) -> None:
        print(f"{label:<32}" + "".join(f"{value:>13}" for value in values))

    def cell(func) -> list[str]:
        return [func(tallies[name]) for name in columns]

    if args.mode == "holdout":
        for k in args.top_k:
            row(f"recall@{k} of hidden tag", cell(lambda t, cutoff=k: f"{t.recall_at(cutoff):.1%}"))
        row("MRR of hidden tag", cell(lambda t: f"{t.mrr:.3f}"))
    else:
        row("recall of applied tags", cell(lambda t: f"{t.recall:.1%}"))

    row("suggestion precision", cell(lambda t: f"{t.precision:.1%}"))
    row("suggestions per entry", cell(lambda t: f"{t.mean_suggestions:.2f}"))
    row("sibling conflicts / entry", cell(lambda t: f"{t.conflict_rate:.2f}"))
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    library = Library()
    status = library.open_library(args.library)
    if not status.success:
        print(f"Could not open library at {args.library}", file=sys.stderr)
        return 1

    config = EmbeddingConfig(
        model=args.model,
        dimensions=args.dimensions,
        device=args.device,
        use_images=not args.no_images,
    )
    index = library.configure_semantic_index(config)
    print(f"Loading {config.model} ({config.dimensions}d) on {config.device}...", file=sys.stderr)
    index.prepare()

    all_tags = library.tags
    tags = index.candidate_tags(all_tags)
    if not tags:
        print("The library has no candidate tags to suggest.", file=sys.stderr)
        return 1

    print(f"Embedding {len(tags)} tag labels...", file=sys.stderr)
    tag_vectors = index.backend.encode([tag_label(tag) for tag in tags])
    facets = tag_facets(all_tags)

    entries = library.get_entries_for_embedding()
    if args.limit:
        entries = entries[: args.limit]

    tallies, hubness_available = evaluate(
        index, library, entries, tags, tag_vectors, facets, args, random.Random(args.seed)
    )
    print_report(args, tallies, hubness_available)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
