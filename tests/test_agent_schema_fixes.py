from __future__ import annotations

import pytest

from contractsentinel.agent.fixes import FixError, apply_fix, run_tests, touches_contract, validate_fix
from contractsentinel.agent.schema import (
    Edit,
    ProposedFix,
    AnalysisError,
    parse_analysis,
    submit_spec,
)


def good_args(**over):
    args = {
        "root_cause": "source_defect",
        "confidence": "high",
        "summary": "The source sent bad statuses.",
        "evidence": ["status had a typo on 7 rows"],
        "impact": "7 rows quarantined.",
        "recommended_actions": ["Tell the source owner"],
    }
    args.update(over)
    return args


# ---- schema ---------------------------------------------------------------------------


def test_parse_valid_report_without_fix():
    a = parse_analysis(good_args(suspect_commit=" abc1234 "))
    assert a.root_cause == "source_defect" and a.fix is None and a.suspect_commit == "abc1234"
    assert a.evidence == ("status had a typo on 7 rows",)


def test_parse_valid_report_with_fix():
    fix = {"summary": "restore", "edits": [{"path": "src/a.py", "old": "x", "new": "y"}]}
    a = parse_analysis(good_args(root_cause="code_regression", fix=fix))
    assert a.fix.edits == (Edit("src/a.py", "x", "y"),)


@pytest.mark.parametrize(
    "over, fragment",
    [
        ({"root_cause": "gremlins"}, "root_cause"),
        ({"confidence": "certain"}, "confidence"),
        ({"summary": ""}, "'summary' is required"),
        ({"summary": 5}, "must be a string"),
        ({"summary": "x" * 2000}, "too long"),
        ({"evidence": []}, "non-empty list"),
        ({"evidence": "not a list"}, "non-empty list"),
        ({"evidence": ["ok", " "]}, "non-empty strings"),
        ({"evidence": ["x" * 500]}, "too long"),
        ({"recommended_actions": ["a"] * 9}, "too many"),
        ({"fix": "patch it"}, "'fix' must be an object"),
        ({"fix": {"summary": "s", "edits": []}}, "non-empty list"),
        ({"fix": {"summary": "s", "edits": [1]}}, "must be an object"),
        ({"fix": {"summary": "s", "edits": [{"path": "", "old": "a", "new": "b"}]}}, "path"),
        ({"fix": {"summary": "s", "edits": [{"path": "src/a", "old": 1, "new": "b"}]}}, "strings"),
        ({"fix": {"summary": "s", "edits": [{"path": "src/a", "old": "a" * 7000, "new": "b"}]}}, "too long"),
        ({"fix": {"summary": "s", "edits": [{"path": "src/a", "old": "a", "new": "b"}] * 6}}, "too many"),
    ],
)
def test_parse_rejects_bad_reports(over, fragment):
    with pytest.raises(AnalysisError) as exc:
        parse_analysis(good_args(**over))
    assert fragment in str(exc.value)


def test_missing_required_field_is_rejected():
    args = good_args()
    del args["impact"]
    with pytest.raises(AnalysisError):
        parse_analysis(args)


def test_submit_spec_lists_every_root_cause():
    spec = submit_spec()
    assert spec.parameters["properties"]["root_cause"]["enum"][0] == "source_defect"
    assert "fix" in spec.parameters["properties"]


# ---- validate / apply -----------------------------------------------------------------


def fix_of(*edits, summary="s"):
    return ProposedFix(summary, tuple(edits))


def test_valid_code_fix_passes_validation(incident_env):
    fix = fix_of(Edit("src/demo/transform.py", "return value", "return value.strip()"))
    assert validate_fix(fix, "code_regression", incident_env.repo) == []


@pytest.mark.parametrize(
    "path",
    ["README.md", ".github/workflows/ci.yml", "../outside.py", "/etc/passwd", "src/../README.md", "pyproject.toml"],
)
def test_paths_outside_the_allowlist_are_rejected(incident_env, path):
    problems = validate_fix(fix_of(Edit(path, "a", "b")), "code_regression", incident_env.repo)
    assert problems and "allowed paths" in problems[0]


def test_old_text_must_match_exactly_once(incident_env):
    missing = validate_fix(fix_of(Edit("src/demo/transform.py", "nope", "x")), "code_regression", incident_env.repo)
    assert "found 0" in missing[0]
    twice = validate_fix(fix_of(Edit("src/demo/transform.py", "e", "x")), "code_regression", incident_env.repo)
    assert "found" in twice[0] and "exactly once" in twice[0]


def test_missing_file_is_rejected(incident_env):
    problems = validate_fix(fix_of(Edit("src/demo/ghost.py", "a", "b")), "code_regression", incident_env.repo)
    assert "does not exist" in problems[0]


def test_contract_edits_need_the_contract_out_of_date_root_cause(incident_env):
    old = "allowed: [NA, EMEA, APAC, LATAM]"
    assert (incident_env.repo / "contracts" / "orders.yaml").read_text().count(old) == 1
    edit = Edit("contracts/orders.yaml", old, "allowed: [NA, EMEA, APAC, LATAM, MEA]")
    wrong = validate_fix(fix_of(edit), "source_defect", incident_env.repo)
    assert any("contract_out_of_date" in p for p in wrong)
    assert validate_fix(fix_of(edit), "contract_out_of_date", incident_env.repo) == []


def test_new_files_only_under_tests(incident_env):
    ok = validate_fix(fix_of(Edit("tests/test_new.py", "", "x = 1\n")), "code_regression", incident_env.repo)
    assert ok == []
    bad = validate_fix(fix_of(Edit("src/demo/new.py", "", "x = 1\n")), "code_regression", incident_env.repo)
    assert "only new files under tests/" in bad[0]
    exists = validate_fix(fix_of(Edit("tests/test_demo.py", "", "x")), "code_regression", incident_env.repo)
    assert "already exists" in exists[0]


def test_oversized_fix_is_rejected(incident_env):
    big = "x" * 6000
    edits = [Edit(f"tests/test_big{i}.py", "", big) for i in range(4)]
    problems = validate_fix(fix_of(*edits), "code_regression", incident_env.repo)
    assert any("too large" in p for p in problems)


def test_symlink_escape_is_rejected(incident_env, tmp_path):
    import os

    outside = tmp_path / "outside.py"
    outside.write_text("a")
    os.symlink(outside, incident_env.repo / "src" / "demo" / "evil.py")
    problems = validate_fix(fix_of(Edit("src/demo/evil.py", "a", "b")), "code_regression", incident_env.repo)
    assert any("escapes" in p for p in problems)


def test_apply_edits_and_create_files(incident_env):
    fix = fix_of(
        Edit("src/demo/transform.py", "return value", "return value.strip()"),
        Edit("src/demo/transform.py", '"""Dates pass through unchanged."""', '"""Trim."""'),
        Edit("tests/test_new.py", "", "def test_x():\n    assert True\n"),
    )
    changed = apply_fix(fix, incident_env.repo)
    assert changed == ["src/demo/transform.py", "tests/test_new.py"]
    text = (incident_env.repo / "src" / "demo" / "transform.py").read_text()
    assert "value.strip()" in text and "Trim." in text
    assert (incident_env.repo / "tests" / "test_new.py").exists()


def test_apply_raises_when_text_no_longer_matches(incident_env):
    with pytest.raises(FixError):
        apply_fix(fix_of(Edit("src/demo/transform.py", "nope", "x")), incident_env.repo)


def test_touches_contract():
    assert touches_contract(fix_of(Edit("contracts/orders.yaml", "a", "b")))
    assert not touches_contract(fix_of(Edit("src/a.py", "a", "b")))


# ---- run_tests ------------------------------------------------------------------------


def test_run_tests_passes_and_fails(incident_env):
    passed, _ = run_tests(incident_env.repo)
    assert passed
    (incident_env.repo / "tests" / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    passed, output = run_tests(incident_env.repo)
    assert not passed and "test_bad" in output
