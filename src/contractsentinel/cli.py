"""Command line interface.

Exit codes:
  0  batch passed the contract
  1  contract violated (this is the trigger for the investigation agent)
  2  the tool itself failed (bad contract, missing file, connection error)
"""

from __future__ import annotations

import argparse
import os
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

    inv = sub.add_parser(
        "investigate",
        help="Investigate failed batches and open a GitHub Issue or PR.",
        description="Reads run artifacts, investigates each failed batch, and opens an Issue "
        "(or a PR when a tested fix exists). Needs GITHUB_TOKEN and GITHUB_REPOSITORY unless "
        "--dry-run is used.",
    )
    inv.add_argument("--runs", default="runs", help="Directory containing run artifacts.")
    inv.add_argument("--batch", help="Only investigate this batch id.")
    inv.add_argument("--contract", default="contracts/orders.yaml")
    inv.add_argument("--repo-root", default=".", help="Repository checkout the agent may read.")
    inv.add_argument("--provider", help="LLM provider (default: LLM_PROVIDER or openai).")
    inv.add_argument("--no-llm", action="store_true", help="Deterministic report only.")
    inv.add_argument("--dry-run", action="store_true", help="Print the report; change nothing.")

    ev = sub.add_parser(
        "eval",
        help="Score models on scenarios with known answers.",
        description="Runs the investigation agent on built-in broken-batch scenarios, once per "
        "trial per model, and writes a Markdown report. Calls the real LLM, so it costs money "
        "(a few cents per model with the default settings).",
    )
    ev.add_argument("--models", required=True, help="Comma-separated model ids, e.g. gpt-6-luna,gpt-6.1-sol")
    ev.add_argument("--trials", type=int, default=3, help="Runs per scenario per model.")
    ev.add_argument("--scenarios", help="Comma-separated subset (default: all).")
    ev.add_argument("--provider", help="LLM provider (default: LLM_PROVIDER or openai).")
    ev.add_argument("--out", default="eval-results", help="Directory for report.md and results.json.")
    ev.add_argument("--price", action="append", help="MODEL=IN,OUT USD per million tokens. Repeatable.")
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


def _cmd_investigate(args: argparse.Namespace) -> int:
    from contractsentinel.agent.github import GitHubClient
    from contractsentinel.agent.investigate import InvestigationSettings, investigate_runs
    from contractsentinel.llm import LLMConfigError, get_llm_client

    contract = load_contract(args.contract)

    llm = None
    if not args.no_llm:
        try:
            llm = get_llm_client(args.provider)
        except LLMConfigError as exc:
            print(f"warning: {exc} Continuing with the deterministic report only.", file=sys.stderr)

    gh = None
    if not args.dry_run:
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPOSITORY")
        if not token or not repo:
            print("error: GITHUB_TOKEN and GITHUB_REPOSITORY are required (or use --dry-run).", file=sys.stderr)
            return EXIT_ERROR
        gh = GitHubClient(token, repo, api_url=os.getenv("GITHUB_API_URL", "https://api.github.com"))

    run_url = None
    if os.getenv("GITHUB_RUN_ID") and os.getenv("GITHUB_REPOSITORY"):
        server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
        run_url = f"{server}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"

    settings = InvestigationSettings(
        repo_root=Path(args.repo_root).resolve(),
        contract=contract,
        dry_run=args.dry_run,
        run_url=run_url,
    )
    results = investigate_runs(args.runs, settings, llm, gh, only_batch=args.batch)
    if not results:
        print("No run artifacts found to investigate.")
        return EXIT_OK

    for result in results:
        print(f"{result.batch_id}: {result.action}" + (f" {result.url}" if result.url else ""))
        for note in result.notes:
            print(f"  note: {note}")
        if args.dry_run and result.action == "dry_run":
            print()
            print(result.body)
    return EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
    from contractsentinel import evals
    from contractsentinel.llm import get_llm_client

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("error: --models needs at least one model id.", file=sys.stderr)
        return EXIT_ERROR
    if not 1 <= args.trials <= 10:
        print("error: --trials must be between 1 and 10.", file=sys.stderr)
        return EXIT_ERROR
    scenarios = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
    prices = evals.parse_prices(args.price)

    def make_llm(model: str):
        os.environ["OPENAI_MODEL"] = model
        return get_llm_client(args.provider)

    results = evals.run_eval(
        models, make_llm, args.trials, scenarios, progress=lambda line: print(line, flush=True)
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = evals.render_report(results, prices)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(evals.results_json(results), encoding="utf-8")
    print()
    print(report)
    return EXIT_OK


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
        if args.command == "investigate":
            return _cmd_investigate(args)
        if args.command == "eval":
            return _cmd_eval(args)
        return _cmd_run(args)
    except (ContractError, RuntimeError, ValueError, OSError) as exc:  # includes GitHubError
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
