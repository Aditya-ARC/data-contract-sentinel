"""Model evaluation: run the agent on scenarios with known answers and score it.

Each scenario is a small git repository plus a broken batch whose true cause is
known. The agent investigates it with the same tools and loop as production, and
the result is checked deterministically (no model grades another model):

  * did it name the right root cause?
  * did it find the commit that caused the problem (when there is one)?
  * for fixes: does the fix apply, do the tests pass, and does the broken batch
    pass the contract afterwards?
  * did it avoid proposing a fix when none is appropriate, or loosening the
    contract when the source is simply wrong?

Nothing here talks to GitHub. Only the LLM calls cost money.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from contractsentinel.agent.fixes import apply_fix, touches_contract, validate_fix
from contractsentinel.agent.investigate import build_brief
from contractsentinel.agent.loop import AgentConfig, investigate
from contractsentinel.agent.mask import mask_violations
from contractsentinel.agent.tools import ToolBox
from contractsentinel.contract import Contract, load_contract
from contractsentinel.llm.base import LLMClient
from contractsentinel.pipeline import run_batch
from contractsentinel.sample_data import generate_orders, write_samples, write_slash_date_sample

# USD per million tokens (input, output), from OpenAI's public pricing page in Oct 2026.
# Override or extend with --price MODEL=IN,OUT. Unknown models are reported without cost.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}

SCENARIO_NAMES = ("source_defect", "code_regression", "contract_out_of_date", "schema_drift")
GATE_SECONDS = 300


class EvalError(RuntimeError):
    pass


@dataclass
class Scenario:
    name: str
    description: str
    expected_root_cause: str
    expect_fix: str  # "none": no fix is right, "required": a working fix is right, "optional"
    repo: Path
    batch: Path
    artifacts: Path
    suspect_commit: str | None = None  # full hash of the commit that caused the problem


@dataclass
class TrialResult:
    model: str
    scenario: str
    trial: int
    passed: bool
    checks: dict[str, bool]
    root_cause: str | None
    stop_reason: str
    steps: int
    input_tokens: int
    output_tokens: int
    seconds: float
    error: str | None = None
    notes: str = ""


# ---- scenario construction --------------------------------------------------------------


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "src" / "contractsentinel").is_dir() or not (root / "tests").is_dir():
        raise EvalError("evals need a source checkout of the project (src/ and tests/ must exist).")
    return root


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=Dev", "-c", "user.email=dev@example.com", "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


_SLASH_BLOCK = '''    if "order_date" in out.columns:
        out["order_date"] = out["order_date"].str.replace(
            r"^(\\d{4})/(\\d{2})/(\\d{2})$", r"\\1-\\2-\\3", regex=True
        )

'''


def _with_slash_handling(text: str) -> str:
    """Return transform.py text that converts 2026/10/04 to ISO, adding it if missing."""
    if '"order_date" in out.columns' in text:
        return text
    marker = "    return out\n\n\ndef to_typed"
    if marker not in text:
        raise EvalError("transform.py has an unexpected layout; update evals._with_slash_handling.")
    return text.replace(marker, _SLASH_BLOCK + marker, 1)


def _without_slash_handling(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith('    if "order_date" in out.columns'):
            skipping = True
            continue
        if skipping:
            if line.startswith("        ") or line.strip() == "":
                if line.strip() == "":
                    skipping = False
                continue
            skipping = False
        out.append(line)
    result = "\n".join(out)
    if result == text:
        raise EvalError("could not remove the slash-date handling from transform.py")
    return result


VENDOR_NOTICES = """# Vendor notices

Notices received from the orders vendor, newest first.

- 2026-09-15: The vendor moved its SFTP host. Update credentials before 2026-10-01.
- 2026-08-20: Vendor support hours changed to 08:00 to 18:00 Pacific.
"""

VENDOR_NOTICE_NEW_REGION = """# Vendor notices

Notices received from the orders vendor, newest first.

- 2026-10-01: Starting with the 2026-10-04 export the vendor adds a new sales region code,
  `MEA` (Middle East and Africa), split out of EMEA. This is intentional and confirmed with
  the data platform team. The region field will now contain NA, EMEA, MEA, APAC or LATAM.
- 2026-09-15: The vendor moved its SFTP host. Update credentials before 2026-10-01.
- 2026-08-20: Vendor support hours changed to 08:00 to 18:00 Pacific.
"""


def _base_repo(repo: Path, notices: str) -> None:
    root = _project_root()
    repo.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copytree(root / "src", repo / "src", ignore=ignore)
    (repo / "contracts").mkdir()
    shutil.copy(root / "contracts" / "orders.yaml", repo / "contracts" / "orders.yaml")
    (repo / "tests").mkdir()
    for name in ("conftest.py", "test_transform.py", "test_validator.py", "test_contract.py"):
        shutil.copy(root / "tests" / name, repo / "tests" / name)
    shutil.copy(root / "pyproject.toml", repo / "pyproject.toml")
    (repo / "docs").mkdir()
    (repo / "docs" / "vendor-notices.md").write_text(notices, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n*.egg-info/\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")


def _regress_slash_dates(repo: Path) -> str:
    transform = repo / "src" / "contractsentinel" / "transform.py"
    transform.write_text(_without_slash_handling(transform.read_text(encoding="utf-8")), encoding="utf-8")
    tests = repo / "tests" / "test_transform.py"
    tests.write_text(tests.read_text(encoding="utf-8").replace("2026/10/04", "2026-10-04"), encoding="utf-8")
    return _commit(repo, "Simplify date handling in normalize()\n\nThe contract checks dates anyway.")


def build_scenario(name: str, workdir: Path, contract_path: Path | None = None) -> Scenario:
    """Create the repository, batch file and run artifacts for one scenario."""
    if name not in SCENARIO_NAMES:
        raise EvalError(f"unknown scenario {name!r}. Choose from {', '.join(SCENARIO_NAMES)}.")
    workdir = Path(workdir)
    repo = workdir / "repo"
    batches = workdir / "batches"
    batches.mkdir(parents=True)
    suspect = None

    if name == "source_defect":
        _base_repo(repo, VENDOR_NOTICES)
        _commit(repo, "Initial pipeline")
        _, batch = write_samples(batches)
        description = "The vendor sent seven rows with bad values. The code is fine."
        cause, expect_fix = "source_defect", "none"

    elif name == "code_regression":
        _base_repo(repo, VENDOR_NOTICES)
        transform = repo / "src" / "contractsentinel" / "transform.py"
        transform.write_text(_with_slash_handling(transform.read_text(encoding="utf-8")), encoding="utf-8")
        _commit(repo, "Initial pipeline")
        batch = write_slash_date_sample(batches)
        suspect = _regress_slash_dates(repo)
        description = "A refactor removed slash-date handling, so a valid batch fails the date rule."
        cause, expect_fix = "code_regression", "required"

    elif name == "contract_out_of_date":
        _base_repo(repo, VENDOR_NOTICE_NEW_REGION)
        _commit(repo, "Initial pipeline")
        df = generate_orders(40, "2026-10-04", seed=11, id_start=400001)
        df.loc[df.index[:12], "region"] = "MEA"
        batch = batches / "orders_2026-10-04.csv"
        df.to_csv(batch, index=False)
        description = "The vendor announced a new region code, MEA. The contract has not caught up."
        cause, expect_fix = "contract_out_of_date", "required"

    else:  # schema_drift
        _base_repo(repo, VENDOR_NOTICES)
        _commit(repo, "Initial pipeline")
        df = generate_orders(40, "2026-10-05", seed=13, id_start=500001)
        batch = batches / "orders_2026-10-05.csv"
        df.rename(columns={"customer_id": "cust_id"}).to_csv(batch, index=False)
        description = "The vendor renamed customer_id to cust_id. No notice, no code change."
        cause, expect_fix = "schema_drift", "optional"

    contract = load_contract(contract_path or repo / "contracts" / "orders.yaml")
    result = run_batch(batch, contract, None, workdir / "runs", dry_run=True)
    if not result.needs_agent:
        raise EvalError(f"scenario {name} did not trigger the agent; the batch passed the contract.")

    scenario = Scenario(
        name=name,
        description=description,
        expected_root_cause=cause,
        expect_fix=expect_fix,
        repo=repo,
        batch=batch,
        artifacts=result.artifacts_dir,
        suspect_commit=suspect,
    )
    if name == "code_regression":
        # Sanity check: the code before the regression must have handled this batch.
        probe = workdir / "probe"
        _git(repo, "worktree", "add", "-q", str(probe), f"{suspect}~1")
        try:
            if not batch_passes(probe, batch):
                raise EvalError("baseline code does not pass the slash-date batch; evals are not valid.")
        finally:
            _git(repo, "worktree", "remove", "--force", str(probe))
    return scenario


# ---- oracles -----------------------------------------------------------------------------


def _env_for(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    return env


def batch_passes(repo: Path, batch: Path) -> bool:
    """Run the batch through the repo's own code and contract. True means it passes."""
    with tempfile.TemporaryDirectory() as tmp:
        code = (
            "import sys; from contractsentinel.cli import main; sys.exit(main(sys.argv[1:]))"
        )
        result = subprocess.run(
            [
                sys.executable, "-c", code, "run", str(batch),
                "--contract", str(repo / "contracts" / "orders.yaml"),
                "--artifacts", tmp, "--dry-run",
            ],
            cwd=repo,
            env=_env_for(repo),
            capture_output=True,
            text=True,
            timeout=GATE_SECONDS,
        )
    return result.returncode == 0


def default_gate(repo: Path) -> bool:
    """The scenario repo's own test suite, run against the scenario repo's own code."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"],
            cwd=repo,
            env=_env_for(repo),
            capture_output=True,
            text=True,
            timeout=GATE_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _commit_matches(claimed: str | None, actual: str | None) -> bool:
    if not claimed or not actual:
        return False
    claimed = claimed.strip().lower()
    return len(claimed) >= 7 and actual.lower().startswith(claimed)


# ---- one trial ---------------------------------------------------------------------------


def cost_usd(model: str, input_tokens: int, output_tokens: int, prices=None) -> float | None:
    table = prices if prices is not None else PRICES_PER_MILLION
    if model not in table:
        return None
    p_in, p_out = table[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def run_trial(
    scenario: Scenario,
    llm: LLMClient,
    trial: int,
    workdir: Path,
    config: AgentConfig | None = None,
    gate: Callable[[Path], bool] = default_gate,
) -> TrialResult:
    """Investigate a fresh copy of the scenario and score the result."""
    repo = Path(workdir) / "trial_repo"
    shutil.copytree(scenario.repo, repo)
    summary = json.loads((scenario.artifacts / "summary.json").read_text(encoding="utf-8"))
    violations = json.loads((scenario.artifacts / "violations.json").read_text(encoding="utf-8"))
    contract: Contract = load_contract(repo / "contracts" / "orders.yaml")
    brief = build_brief(summary, mask_violations(violations, contract))

    started = time.monotonic()
    outcome = investigate(llm, ToolBox(repo, scenario.artifacts, contract), brief, config or AgentConfig())
    seconds = time.monotonic() - started

    analysis = outcome.analysis
    checks: dict[str, bool] = {"diagnosed": analysis is not None}
    notes: list[str] = []
    if analysis is not None:
        checks["root_cause"] = analysis.root_cause == scenario.expected_root_cause
        if scenario.suspect_commit:
            checks["suspect_commit"] = _commit_matches(analysis.suspect_commit, scenario.suspect_commit)

        fix = analysis.fix
        if scenario.expect_fix == "none":
            checks["no_fix_proposed"] = fix is None
        elif scenario.expect_fix == "optional":
            checks["no_contract_loosening"] = fix is None or not touches_contract(fix)
        else:  # required
            checks["fix_proposed"] = fix is not None
            if fix is not None:
                problems = validate_fix(fix, analysis.root_cause, repo)
                checks["fix_valid"] = not problems
                tests_ok = fixes_batch = False
                if not problems:
                    apply_fix(fix, repo)
                    tests_ok = gate(repo)
                    fixes_batch = batch_passes(repo, scenario.batch)
                checks["fix_tests_pass"] = tests_ok
                checks["fix_fixes_batch"] = fixes_batch
                if problems:
                    notes.append("; ".join(problems)[:200])
    return TrialResult(
        model=llm.model,
        scenario=scenario.name,
        trial=trial,
        passed=all(checks.values()),
        checks=checks,
        root_cause=analysis.root_cause if analysis else None,
        stop_reason=outcome.stop_reason,
        steps=outcome.steps,
        input_tokens=outcome.usage.input_tokens,
        output_tokens=outcome.usage.output_tokens,
        seconds=round(seconds, 1),
        error=outcome.error,
        notes=" | ".join(notes),
    )


# ---- running and reporting ---------------------------------------------------------------


def run_eval(
    models: list[str],
    make_llm: Callable[[str], LLMClient],
    trials: int,
    scenarios: list[str] | None = None,
    config: AgentConfig | None = None,
    gate: Callable[[Path], bool] = default_gate,
    progress: Callable[[str], None] | None = None,
) -> list[TrialResult]:
    names = scenarios or list(SCENARIO_NAMES)
    results: list[TrialResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        built = {n: build_scenario(n, Path(tmp) / "scenarios" / n) for n in names}
        for model in models:
            llm = make_llm(model)
            for name in names:
                for t in range(1, trials + 1):
                    trial_dir = Path(tmp) / "trials" / model.replace("/", "_") / name / str(t)
                    trial_dir.mkdir(parents=True)
                    result = run_trial(built[name], llm, t, trial_dir, config, gate)
                    results.append(result)
                    if progress:
                        progress(f"{model} {name} #{t}: {'PASS' if result.passed else 'FAIL'} {result.checks}")
    return results


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}" if value < 0.1 else f"${value:.2f}"


def render_report(results: list[TrialResult], prices=None) -> str:
    models = list(dict.fromkeys(r.model for r in results))
    scenarios = list(dict.fromkeys(r.scenario for r in results))
    lines = ["# Model evaluation", ""]

    lines += [
        "Scored deterministically against scenarios with known answers. "
        "A trial passes only if every check passes.",
        "",
        "| Model | Passed | Right root cause | Avg steps | Avg tokens (in / out) | Avg seconds | Avg cost per incident |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        rs = [r for r in results if r.model == m]
        passed = sum(r.passed for r in rs)
        cause = sum(r.checks.get("root_cause", False) for r in rs)
        costs = [cost_usd(m, r.input_tokens, r.output_tokens, prices) for r in rs]
        avg_cost = None if any(c is None for c in costs) else statistics.mean(costs)
        lines.append(
            f"| `{m}` | {passed}/{len(rs)} | {cause}/{len(rs)} | "
            f"{statistics.mean(r.steps for r in rs):.1f} | "
            f"{statistics.mean(r.input_tokens for r in rs):,.0f} / {statistics.mean(r.output_tokens for r in rs):,.0f} | "
            f"{statistics.mean(r.seconds for r in rs):.0f} | {_fmt_cost(avg_cost)} |"
        )

    lines += ["", "## Passes by scenario", "", "| Scenario | " + " | ".join(f"`{m}`" for m in models) + " |"]
    lines.append("|---|" + "---|" * len(models))
    for s in scenarios:
        row = [s]
        for m in models:
            rs = [r for r in results if r.model == m and r.scenario == s]
            row.append(f"{sum(r.passed for r in rs)}/{len(rs)}")
        lines.append("| " + " | ".join(row) + " |")

    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            failed = [k for k, v in r.checks.items() if not v] or [f"stopped: {r.stop_reason}"]
            detail = [x for x in (r.error and r.error[:120], r.notes) if x]
            note = f" ({'; '.join(detail)})" if detail else ""
            if "diagnosed" in failed:
                note = f" (stop reason: {r.stop_reason}){note if r.error else ''}"
            lines.append(
                f"- `{r.model}` {r.scenario} #{r.trial}: failed {', '.join(failed)}; "
                f"root cause given: {r.root_cause}{note}"
            )
    lines += [
        "",
        "Costs use the prices in `evals.PRICES_PER_MILLION` and are estimates. "
        "Small trial counts show tendencies, not guarantees.",
        "",
    ]
    return "\n".join(lines)


def results_json(results: list[TrialResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)


def parse_prices(items: list[str] | None) -> dict[str, tuple[float, float]]:
    table = dict(PRICES_PER_MILLION)
    for item in items or []:
        match = re.fullmatch(r"([^=\s]+)=([0-9.]+),([0-9.]+)", item)
        if not match:
            raise EvalError(f"--price must look like MODEL=IN,OUT, got {item!r}")
        table[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return table
