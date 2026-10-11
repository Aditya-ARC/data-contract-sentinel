from __future__ import annotations

import pytest

from contractsentinel import evals
from contractsentinel.agent.schema import SUBMIT_TOOL_NAME
from contractsentinel.cli import EXIT_ERROR, main
from contractsentinel.llm.scripted import ScriptedLLMClient, tool_response

TRANSFORM = "src/contractsentinel/transform.py"
RESTORE_MARKER = "    return out\n\n\ndef to_typed"
REGION_OLD = "allowed: [NA, EMEA, APAC, LATAM]"


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory):
    root = tmp_path_factory.mktemp("scn")
    return {n: evals.build_scenario(n, root / n) for n in evals.SCENARIO_NAMES}


def report(root_cause, fix=None, suspect=None):
    r = {
        "root_cause": root_cause,
        "confidence": "high",
        "summary": "s",
        "evidence": ["e"],
        "impact": "i",
        "recommended_actions": ["a"],
    }
    if fix:
        r["fix"] = fix
    if suspect:
        r["suspect_commit"] = suspect
    return r


def trial(scenario, answer, tmp_path, gate=lambda repo: True):
    llm = ScriptedLLMClient([tool_response(SUBMIT_TOOL_NAME, answer)], model="scripted")
    return evals.run_trial(scenario, llm, 1, tmp_path, gate=gate)


def slash_fix():
    return {
        "summary": "restore",
        "edits": [{"path": TRANSFORM, "old": RESTORE_MARKER, "new": evals._SLASH_BLOCK + RESTORE_MARKER}],
    }


def region_fix(update_test=True):
    edits = [{"path": "contracts/orders.yaml", "old": REGION_OLD, "new": "allowed: [NA, EMEA, MEA, APAC, LATAM]"}]
    if update_test:  # a test pins the exact list, so a complete fix updates it too
        edits.append(
            {
                "path": "tests/test_contract.py",
                "old": 'allowed == ("NA", "EMEA", "APAC", "LATAM")',
                "new": 'allowed == ("NA", "EMEA", "MEA", "APAC", "LATAM")',
            }
        )
    return {"summary": "add MEA", "edits": edits}


# ---- scenarios are valid ---------------------------------------------------------------


def test_every_scenario_triggers_the_agent_and_has_artifacts(scenarios):
    for s in scenarios.values():
        assert (s.artifacts / "summary.json").exists() and (s.artifacts / "violations.json").exists()


def test_regression_scenario_is_a_real_regression(scenarios):
    s = scenarios["code_regression"]
    assert not evals.batch_passes(s.repo, s.batch)  # the oracle runs the scenario's own code
    log = evals._git(s.repo, "log", "--format=%H %s")
    assert s.suspect_commit in log and "Simplify date handling" in log
    assert evals.default_gate(s.repo)  # the regression slipped past the tests, as in real life


def test_unknown_scenario_is_rejected(tmp_path):
    with pytest.raises(evals.EvalError):
        evals.build_scenario("nope", tmp_path / "x")


# ---- scoring: right answers pass, wrong ones fail -------------------------------------


def test_regression_perfect_answer_passes_with_the_real_gate(scenarios, tmp_path):
    s = scenarios["code_regression"]
    r = trial(s, report("code_regression", slash_fix(), s.suspect_commit[:7]), tmp_path, gate=evals.default_gate)
    assert r.passed, r.checks


def test_regression_wrong_cause_fails(scenarios, tmp_path):
    r = trial(scenarios["code_regression"], report("source_defect"), tmp_path)
    assert not r.passed and not r.checks["root_cause"] and not r.checks["fix_proposed"]


def test_regression_wrong_commit_fails(scenarios, tmp_path):
    r = trial(scenarios["code_regression"], report("code_regression", slash_fix(), "abc1234"), tmp_path)
    assert r.checks["root_cause"] and not r.checks["suspect_commit"] and not r.passed


def test_fix_that_does_not_fix_the_batch_fails(scenarios, tmp_path):
    useless = {
        "summary": "comment",
        "edits": [{"path": TRANSFORM, "old": RESTORE_MARKER, "new": "    # TODO\n" + RESTORE_MARKER}],
    }
    s = scenarios["code_regression"]
    r = trial(s, report("code_regression", useless, s.suspect_commit[:7]), tmp_path)
    assert r.checks["fix_valid"] and r.checks["fix_tests_pass"]
    assert not r.checks["fix_fixes_batch"] and not r.passed


def test_fix_that_breaks_the_tests_fails(scenarios, tmp_path):
    s = scenarios["code_regression"]
    r = trial(s, report("code_regression", slash_fix(), s.suspect_commit[:7]), tmp_path, gate=lambda repo: False)
    assert r.checks["fix_fixes_batch"] and not r.checks["fix_tests_pass"] and not r.passed


def test_invalid_fix_is_flagged(scenarios, tmp_path):
    bad = {"summary": "x", "edits": [{"path": "README.md", "old": "a", "new": "b"}]}
    s = scenarios["code_regression"]
    r = trial(s, report("code_regression", bad, s.suspect_commit[:7]), tmp_path)
    assert not r.checks["fix_valid"] and not r.checks["fix_tests_pass"]
    assert "allowed paths" in r.notes


def test_source_defect_correct_answer_is_no_fix(scenarios, tmp_path):
    assert trial(scenarios["source_defect"], report("source_defect"), tmp_path).passed


def test_source_defect_with_a_fix_fails(scenarios, tmp_path):
    r = trial(scenarios["source_defect"], report("source_defect", region_fix()), tmp_path)
    assert not r.checks["no_fix_proposed"] and not r.passed


def test_source_defect_misdiagnosed_as_regression_fails(scenarios, tmp_path):
    assert not trial(scenarios["source_defect"], report("code_regression"), tmp_path).passed


def test_out_of_date_contract_fix_passes_with_the_real_gate(scenarios, tmp_path):
    r = trial(scenarios["contract_out_of_date"], report("contract_out_of_date", region_fix()), tmp_path, gate=evals.default_gate)
    assert r.passed, r.checks


def test_contract_only_fix_fails_when_a_test_pins_the_old_list(scenarios, tmp_path):
    r = trial(
        scenarios["contract_out_of_date"],
        report("contract_out_of_date", region_fix(update_test=False)),
        tmp_path,
        gate=evals.default_gate,
    )
    assert r.checks["fix_valid"] and r.checks["fix_fixes_batch"]
    assert not r.checks["fix_tests_pass"] and not r.passed


def test_out_of_date_contract_called_a_source_defect_fails(scenarios, tmp_path):
    assert not trial(scenarios["contract_out_of_date"], report("source_defect"), tmp_path).passed


def test_contract_edit_with_the_wrong_cause_is_invalid(scenarios, tmp_path):
    r = trial(scenarios["contract_out_of_date"], report("source_defect", region_fix()), tmp_path)
    assert not r.checks["root_cause"] and not r.checks["fix_valid"]


def test_schema_drift_without_a_fix_passes(scenarios, tmp_path):
    assert trial(scenarios["schema_drift"], report("schema_drift"), tmp_path).passed


def test_schema_drift_must_not_loosen_the_contract(scenarios, tmp_path):
    loosen = {
        "summary": "x",
        "edits": [{"path": "contracts/orders.yaml", "old": REGION_OLD, "new": "allowed: [NA]"}],
    }
    r = trial(scenarios["schema_drift"], report("schema_drift", loosen), tmp_path)
    assert not r.checks["no_contract_loosening"] and not r.passed


def test_no_conclusion_fails_cleanly(scenarios, tmp_path):
    llm = ScriptedLLMClient([])  # raises on the first call
    r = evals.run_trial(scenarios["source_defect"], llm, 1, tmp_path)
    assert not r.passed and r.stop_reason == "llm_error" and r.error


def test_trials_do_not_contaminate_each_other(scenarios, tmp_path):
    s = scenarios["code_regression"]
    trial(s, report("code_regression", slash_fix(), s.suspect_commit[:7]), tmp_path)
    assert not evals.batch_passes(s.repo, s.batch)  # the shared scenario repo is untouched


# ---- reporting --------------------------------------------------------------------------


def make_result(model, scenario, passed, i=1000, o=100):
    return evals.TrialResult(model, scenario, 1, passed, {"root_cause": passed}, "source_defect", "submitted", 5, i, o, 10.0)


def test_report_summarizes_models_and_failures():
    rs = [make_result("gpt-6-luna", "a", True), make_result("gpt-6-luna", "b", False), make_result("mystery", "a", True)]
    text = evals.render_report(rs)
    assert "`gpt-6-luna` | 1/2" in text and "`mystery` | 1/1" in text
    assert "n/a" in text and "## Failures" in text and "b #1" in text


def test_cost_arithmetic():
    assert evals.cost_usd("gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert evals.cost_usd("gpt-6.1-sol", 30_000, 2_000) == pytest.approx(0.08)
    assert evals.cost_usd("unknown", 1, 1) is None


def test_price_overrides_and_validation():
    assert evals.parse_prices(["x=1,2"])["x"] == (1.0, 2.0)
    with pytest.raises(evals.EvalError):
        evals.parse_prices(["bad"])


def test_json_output_round_trips():
    import json

    assert json.loads(evals.results_json([make_result("m", "a", True)]))[0]["model"] == "m"


def test_run_eval_loops_models_scenarios_and_trials(tmp_path):
    answers = []

    def make_llm(model):
        class L(ScriptedLLMClient):
            pass

        return L([tool_response(SUBMIT_TOOL_NAME, report("source_defect"))] * 2, model=model)

    seen = []
    results = evals.run_eval(["m1", "m2"], make_llm, 2, ["source_defect"], progress=seen.append)
    assert [(r.model, r.trial) for r in results] == [("m1", 1), ("m1", 2), ("m2", 1), ("m2", 2)]
    assert all(r.passed for r in results) and len(seen) == 4


def test_cli_rejects_bad_arguments(capsys):
    assert main(["eval", "--models", " , "]) == EXIT_ERROR
    assert main(["eval", "--models", "m", "--trials", "0"]) == EXIT_ERROR
    assert main(["eval", "--models", "m", "--trials", "99"]) == EXIT_ERROR


def test_failure_lines_say_why():
    stopped = evals.TrialResult("m", "a", 1, False, {"diagnosed": False}, None, "max_steps", 12, 1, 1, 1.0)
    noted = evals.TrialResult("m", "b", 1, False, {"fix_tests_pass": False}, "x", "submitted", 3, 1, 1, 1.0, notes="why")
    text = evals.render_report([stopped, noted])
    assert "stop reason: max_steps" in text and "(why)" in text


def test_report_counts_trials_where_the_warning_was_sent():
    a = evals.TrialResult("m", "a", 1, True, {"root_cause": True}, "x", "submitted", 5, 1, 1, 1.0, warned=True)
    b = evals.TrialResult("m", "a", 2, True, {"root_cause": True}, "x", "submitted", 5, 1, 1, 1.0)
    assert "step warning was sent in 1 of 2 trials" in evals.render_report([a, b])
