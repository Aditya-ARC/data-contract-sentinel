from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from contractsentinel.contract import Contract, load_contract, parse_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "orders.yaml"

BASE_ROW = {
    "order_id": "O100001",
    "customer_id": "C12345",
    "order_date": "2026-10-01",
    "amount": "10.50",
    "status": "pending",
    "region": "NA",
}


@pytest.fixture(scope="session")
def contract() -> Contract:
    return load_contract(CONTRACT_PATH)


@pytest.fixture()
def raw_contract() -> dict:
    """A fresh, mutable dict of the real contract file."""
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def build_contract(raw_contract):
    def _build(mutate=None) -> Contract:
        if mutate:
            mutate(raw_contract)
        return parse_contract(raw_contract)

    return _build


@pytest.fixture()
def make_df():
    """make_df({}, {"customer_id": ""}) -> frame with unique order ids and overrides per row."""

    def _make(*overrides: dict) -> pd.DataFrame:
        rows = []
        for i, override in enumerate(overrides or ({},)):
            row = dict(BASE_ROW)
            row["order_id"] = f"O{100001 + i}"
            row.update(override)
            rows.append(row)
        return pd.DataFrame(rows)

    return _make


# ---- fixtures for the investigation agent ---------------------------------------------

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from contractsentinel.agent.github import GitHubError  # noqa: E402
from contractsentinel.pipeline import run_batch  # noqa: E402
from contractsentinel.sample_data import write_samples  # noqa: E402

TRANSFORM_V1 = '''import re


def normalize_date(value):
    """Convert 2026/10/04 to 2026-10-04."""
    return re.sub(r"^(\\d{4})/(\\d{2})/(\\d{2})$", r"\\1-\\2-\\3", value)
'''

TRANSFORM_V2 = '''def normalize_date(value):
    """Dates pass through unchanged."""
    return value
'''

REGRESSION_TEST = '''import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo.transform import normalize_date


def test_slash_dates_are_normalized():
    assert normalize_date("2026/10/04") == "2026-10-04"
'''


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture()
def incident_env(tmp_path, contract):
    """A git repo with a bare origin, two commits, and real run artifacts for a bad batch."""
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Dev")
    git(repo, "config", "user.email", "dev@example.com")

    (repo / "src" / "demo").mkdir(parents=True)
    (repo / "src" / "demo" / "transform.py").write_text(TRANSFORM_V1)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_demo.py").write_text("def test_ok():\n    assert True\n")
    (repo / "contracts").mkdir()
    shutil.copy(CONTRACT_PATH, repo / "contracts" / "orders.yaml")
    (repo / ".gitignore").write_text("runs/\n__pycache__/\n*.pyc\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Initial code")

    (repo / "src" / "demo" / "transform.py").write_text(TRANSFORM_V2)
    git(repo, "commit", "-qam", "Simplify date handling")
    suspect = git(repo, "rev-parse", "--short", "HEAD")

    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")

    _, bad = write_samples(tmp_path / "landing")
    result = run_batch(bad, contract, None, repo / "runs", dry_run=True)
    return SimpleNamespace(
        repo=repo,
        remote=remote,
        runs=repo / "runs",
        batch_dir=result.artifacts_dir,
        batch_id=result.batch_id,
        suspect=suspect,
    )


class FakeGitHub:
    """Records what the agent would do on GitHub."""

    def __init__(self, existing=None, pr_error: GitHubError | None = None):
        self.existing = existing
        self.pr_error = pr_error
        self.issues: list[dict] = []
        self.prs: list[dict] = []

    def find_open_incident(self, marker, label):
        return self.existing

    def default_branch(self):
        return "main"

    def create_issue(self, title, body, labels):
        self.issues.append({"title": title, "body": body, "labels": labels})
        return {"html_url": f"https://example.test/issues/{len(self.issues)}", "number": len(self.issues)}

    def create_pull_request(self, title, body, head, base, labels):
        if self.pr_error:
            raise self.pr_error
        self.prs.append({"title": title, "body": body, "head": head, "base": base, "labels": labels})
        return {"html_url": f"https://example.test/pull/{len(self.prs)}", "number": len(self.prs)}


@pytest.fixture()
def fake_github():
    return FakeGitHub()
