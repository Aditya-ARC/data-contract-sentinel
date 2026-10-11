"""Command line interface.

Exit codes:
  0  batch passed the contract
  1  contract violated (this is the trigger for the investigation agent)
  2  the tool itself failed (bad contract, missing file, connection error)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contractsentinel.adapters import WAREHOUSES, get_adapter
from contractsentinel.contract import ContractError, load_contract
from contractsentinel.pipeline import RunResult, run_batch
from contractsentinel.sample_data import write_samples
from contractsentinel.summary import render_summary

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contractsentinel", description="Contract-first data pipeline guard."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("make-sample", help="Write a clean and a defective sample batch.")
    sample.add_argument("--out", default="data/landing", help="Output directory.")
    sample.add_argument("--rows", type=int, default=40, help="Rows per batch (minimum 20).")
    sample.add_argument("--seed", type=int, default=7)

    run = sub.add_parser("run", help="Validate and load one batch file.")
    run.add_argument("batch", help="Path to the batch CSV.")
    run.add_argument("--contract", default="contracts/orders.yaml")
    run.add_argument("--warehouse", choices=WAREHOUSES, default="duckdb")
    run.add_argument("--db", default="data/warehouse.duckdb", help="DuckDB file path.")
    run.add_argument("--artifacts", default="runs", help="Where to write run artifacts.")
    run.add_argument("--dry-run", action="store_true", help="Validate only; do not load.")

    summarize = sub.add_parser("summarize", help="Print a Markdown summary of run artifacts.")
    summarize.add_argument("--runs", default="runs", help="Directory containing run artifacts.")
    return parser


def _print_result(result: RunResult) -> None:
    v = result.validation
    print(f"batch:      {result.batch_id}")
    print(f"status:     {result.status}")
    print(f"decision:   {result.decision.action} - {result.decision.reason}")
    loaded = "" if result.dry_run else f", {result.rows_loaded} loaded"
    print(
        f"rows:       {result.rows_total} total, {v.invalid_row_count} invalid"
        f"{loaded}, {result.rows_quarantined} quarantined"
    )
    if result.dry_run:
        print("mode:       dry run, nothing was loaded")
    if v.violations:
        print("violations:")
        for item in v.violations:
            print(f"  [{item.severity}] {item.rule_id} x{item.count}: {item.description}")
    print(f"artifacts:  {result.artifacts_dir}")
    if result.needs_agent:
        print("agent:      contract violated, investigation should be triggered")


def _cmd_make_sample(args: argparse.Namespace) -> int:
    good, bad = write_samples(args.out, rows=args.rows, seed=args.seed)
    print(f"clean batch:     {good}")
    print(f"defective batch: {bad}")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    contract = load_contract(args.contract)
    if not Path(args.batch).is_file():
        print(f"error: batch file not found: {args.batch}", file=sys.stderr)
        return EXIT_ERROR

    if args.dry_run:
        result = run_batch(args.batch, contract, None, args.artifacts, dry_run=True)
    else:
        with get_adapter(args.warehouse, args.db) as adapter:
            result = run_batch(args.batch, contract, adapter, args.artifacts)
    _print_result(result)
    return EXIT_VIOLATIONS if result.needs_agent else EXIT_OK


def _cmd_summarize(args: argparse.Namespace) -> int:
    print(render_summary(args.runs), end="")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "make-sample":
            return _cmd_make_sample(args)
        if args.command == "summarize":
            return _cmd_summarize(args)
        return _cmd_run(args)
    except (ContractError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
