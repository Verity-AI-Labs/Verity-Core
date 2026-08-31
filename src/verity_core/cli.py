"""Command line interface for verity-core.

A convenience layer, not the primary interface: the four tools import the Python API
directly. What the CLI is for is the work around an audit — checking that a corpus
directory parses, seeing which config layer set a value, aggregating a finished run into
a report — none of which should require writing a script.

Human-readable summaries go to stdout, logs to stderr, so ``--json`` output stays safe
to pipe regardless of log level.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity_core import __version__
from verity_core.adapters import ManifestError, load_env
from verity_core.config import resolve_config
from verity_core.corpus import CorpusError, corpus_stats, load_corpus, load_manifest_file
from verity_core.logs import configure_logging
from verity_core.reporting import (
    DEFAULT_DEFECT_THRESHOLD,
    DEFAULT_DIRECTION,
    build_corpus_report,
)
from verity_core.runner import SandboxError
from verity_core.scorecard import Scorecard, scorecard_path

logger = logging.getLogger(__name__)

PROGRAM = "verity-core"
LOG_LEVELS = ("debug", "info", "warning", "error")

EXIT_OK = 0
EXIT_ERROR = 1

# The one axis verity-core can measure without the tools: does the environment's own
# verifier accept its own reference solution? Everything else on the rubric belongs to
# Verity-RedTeam, Verity-Signal, Verity-Clean, and Verity-Stable, which fill in the rest
# of the scorecard.
GOLD_AXIS = "V1"
TOOL_NAME = "verity-core"

# Scored as a defect indicator, not a quality score: 1.0 means the environment rejected
# its own reference solution, which is unambiguously broken. This matches the polarity
# the reporting layer flags on (see reporting.DEFAULT_DIRECTION), so that a corpus of
# healthy environments reports a defect rate near zero rather than near one.
GOLD_FAILS = 1.0
GOLD_PASSES = 0.0

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Inspect corpora, audit single environments, and aggregate results.",
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="warning",
        help="verity_core log verbosity, written to stderr (default: warning)",
    )
    subcommands = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    corpus = subcommands.add_parser(
        "corpus", help="load a corpus directory and summarize it", description="Summarize a corpus."
    )
    corpus.add_argument("path", type=Path, help="directory of YAML manifest files")
    corpus.add_argument("--domain", action="append", help="only this domain (repeatable)")
    corpus.add_argument("--format", action="append", help="only this format (repeatable)")
    corpus.add_argument(
        "--no-recursive", action="store_true", help="do not descend into subdirectories"
    )
    corpus.add_argument(
        "--lenient",
        action="store_true",
        help="skip invalid manifests instead of failing on the first one",
    )
    corpus.add_argument("--list", action="store_true", help="also list every environment id")
    corpus.add_argument("--json", action="store_true", help="emit JSON instead of text")
    corpus.set_defaults(handler=_cmd_corpus)

    audit = subcommands.add_parser(
        "audit",
        help="audit one environment and write a scorecard",
        description=(
            "Audit a single environment. Only the axes verity-core can measure by itself "
            "are filled in; the four tools fill in the rest."
        ),
    )
    audit.add_argument("manifest", type=Path, help="a YAML manifest file")
    audit.add_argument("--results-dir", type=Path, help="write the scorecard here")
    audit.add_argument(
        "--skip-gold",
        action="store_true",
        help="validate the manifest only, without starting a container",
    )
    audit.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    audit.set_defaults(handler=_cmd_audit)

    report = subcommands.add_parser(
        "report",
        help="aggregate scorecards into a corpus report",
        description="Aggregate a results directory into a corpus-level defect report.",
    )
    report.add_argument("results_dir", type=Path, help="directory of scorecard JSON files")
    report.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DEFECT_THRESHOLD,
        help=f"axis score at which an environment is flagged (default: {DEFAULT_DEFECT_THRESHOLD})",
    )
    report.add_argument(
        "--direction",
        choices=("above", "below"),
        default=DEFAULT_DIRECTION,
        help="whether a high or a low score is the defect (default: %(default)s)",
    )
    report.add_argument("--output", type=Path, help="also write corpus-report.{json,md} here")
    report.add_argument("--top", type=int, default=10, help="most-flagged environments to list")
    report.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    report.set_defaults(handler=_cmd_report)

    config = subcommands.add_parser(
        "config",
        help="print the resolved configuration and where each value came from",
        description="Show the effective configuration, annotated by source layer.",
    )
    config.add_argument("--path", type=Path, help="config file to load instead of discovering one")
    config.add_argument("--json", action="store_true", help="emit JSON instead of text")
    config.set_defaults(handler=_cmd_config)

    return parser


def _emit(payload: dict[str, Any] | list[Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_corpus(args: argparse.Namespace) -> int:
    entries = load_corpus(
        args.path,
        domain=args.domain,
        format=args.format,
        recursive=not args.no_recursive,
        strict=not args.lenient,
    )
    stats = corpus_stats(entries)

    if args.json:
        payload: dict[str, Any] = {"path": str(args.path), "stats": stats.to_dict()}
        if args.list:
            payload["environments"] = [str(entry["id"]) for entry in entries]
        _emit(payload)
        return EXIT_OK

    print(f"Corpus: {args.path}")
    print(f"Environments: {stats.total}")
    print(f"With gold solutions: {stats.with_gold} ({stats.gold_coverage:.0%})")
    _print_counts("By domain", stats.by_domain)
    _print_counts("By format", stats.by_format)
    if args.list:
        print("\nEnvironments:")
        for entry in entries:
            print(f"  {entry['id']}  [{entry['format']}]")
    return EXIT_OK


def _print_counts(title: str, counts: dict[str, int]) -> None:
    print(f"\n{title}:")
    if not counts:
        print("  (none)")
        return
    width = max(len(name) for name in counts)
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name.ljust(width)}  {count}")


def _cmd_audit(args: argparse.Namespace) -> int:
    entries = load_manifest_file(args.manifest)
    if not entries:
        print(f"{args.manifest}: no manifest entries found", file=sys.stderr)
        return EXIT_ERROR
    if len(entries) > 1:
        print(
            f"{args.manifest}: expected one environment, found {len(entries)}; "
            "use the batch runner for multiple environments",
            file=sys.stderr,
        )
        return EXIT_ERROR

    entry = entries[0]
    scorecard = _audit_entry(entry, skip_gold=args.skip_gold)

    if args.results_dir is not None:
        scorecard.to_json(scorecard_path(args.results_dir, scorecard.env_id))

    if args.json:
        _emit(scorecard.to_dict())
    else:
        print(scorecard.to_markdown())
    return EXIT_OK


def _audit_entry(entry: dict[str, Any], *, skip_gold: bool) -> Scorecard:
    """Measure what verity-core can measure on its own and return a scorecard."""
    env = load_env(entry)
    spec = env.spec()
    scorecard = Scorecard(
        env_id=spec.id,
        metadata={
            "domain": spec.domain,
            "format": str(entry.get("format", "")),
            "source": spec.source,
            "commit": spec.commit,
            "audited_by": TOOL_NAME,
        },
    )

    gold = env.gold_solution()
    if gold is None:
        scorecard.metadata["gold_solution"] = "absent"
        logger.warning("no gold solution env_id=%s: %s left unscored", spec.id, GOLD_AXIS)
        env.close()
        return scorecard

    scorecard.metadata["gold_solution"] = "present"
    if skip_gold:
        scorecard.metadata["gold_check"] = "skipped"
        env.close()
        return scorecard

    # A fresh reset before verifying keeps this trial independent of anything the
    # environment's own setup left behind.
    with env:
        result = env.verify(gold)

    if not result.verdict:
        logger.warning(
            "gold solution rejected by its own verifier env_id=%s reward=%.3f",
            spec.id,
            result.reward,
        )

    scorecard.set_axis(
        GOLD_AXIS,
        value=GOLD_PASSES if result.verdict else GOLD_FAILS,
        tool=TOOL_NAME,
        evidence={
            "gold_verdict": result.verdict,
            "gold_reward": result.reward,
            "verifier_logs": result.verifier_logs[-2000:],
            "verifier_state": result.verifier_state,
        },
        notes=(
            "1.0 means the environment's own verifier rejected its own reference "
            "solution; 0.0 means it accepted it"
        ),
    )
    return scorecard


def _cmd_report(args: argparse.Namespace) -> int:
    report = build_corpus_report(
        args.results_dir,
        threshold=args.threshold,
        direction=args.direction,
        output_dir=args.output,
    )
    if report.total_scorecards == 0:
        print(f"{args.results_dir}: no scorecards found", file=sys.stderr)

    if args.json:
        _emit(report.to_dict())
    else:
        print(report.to_markdown(top_n=args.top), end="")
    return EXIT_OK


def _cmd_config(args: argparse.Namespace) -> int:
    resolved = resolve_config(args.path)

    if args.json:
        _emit(resolved.to_dict())
        return EXIT_OK

    origin = resolved.path or "none found (environment and defaults only)"
    print(f"Config file: {origin}")
    print("Precedence: file > environment > defaults\n")

    rows = resolved.describe()
    name_width = max(len(name) for name, _, _ in rows)
    value_width = max(len(str(value)) for _, value, _ in rows)
    print(f"{'FIELD'.ljust(name_width)}  {'VALUE'.ljust(value_width)}  FROM")
    for name, value, layer in rows:
        print(f"{name.ljust(name_width)}  {str(value).ljust(value_width)}  {layer}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``verity-core`` console script."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level.upper(), force=True)

    try:
        return int(args.handler(args))
    except (CorpusError, ManifestError, SandboxError, FileNotFoundError, ValueError) as exc:
        # Expected, actionable failures: a bad manifest, an unreachable daemon, a missing
        # directory. Reported as a message rather than a traceback, which would bury the
        # one line the operator needs.
        logger.debug("command failed", exc_info=True)
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print(f"{PROGRAM}: interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
