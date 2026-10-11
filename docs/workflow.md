# Workflow and agent design

This document is the design contract for the investigation agent. Both the pipeline and the agent are built and tested.

## Principle

The contract enforces. The agent investigates and proposes. A human decides.

Validation is deterministic Python driven by `contracts/orders.yaml`. The agent never decides what counts as bad data and is only called after a rule has already failed and the destination has already been protected.

## What triggers the agent

The trigger is the exit code of `contractsentinel run`. Exit code 1 means at least one error-level violation. **Any** error-level violation triggers the agent, including a single bad row that was already quarantined.

| Situation | Pipeline action | Agent runs? |
|---|---|---|
| A row breaks a rule: `not_null`, `type`, `format`, `pattern`, `allowed_values`, `min`, `max`, `scale`, or `unique` | Quarantine the row, or block the batch if more than `max_violation_rate` of rows are invalid | Yes |
| A required column is missing (`missing_column`) or the file is empty (`empty_batch`) | Block the whole batch | Yes |
| An extra column not in the contract (`unexpected_column`) | Warning only, column dropped | No |
| Everything passes | Load all rows | No |
| Tool failure (bad contract, connection error), exit code 2 | Job fails | No, ordinary failed-job alert |

Dedupe rule for the agent phase: the same rule failing on the same batch never opens a second PR or issue.

## End-to-end flow

```
Vendor CSV lands (landing/ or manual dispatch)
  -> Pipeline job (built)
       normalize -> validate against contract -> apply failure policy
         clean rows   -> analytics.orders_clean
         bad rows     -> analytics.orders_quarantine (rules broken + raw source values)
         schema drift -> whole batch blocked
       writes runs/<batch>/summary.json and violations.json
       uploads them as a workflow artifact and writes a job summary
  -> Exit 0: finished
  -> Exit 1: `violations=true` output starts the agent job
       1. read violations, sample rows, transform code, recent git history
       2. classify the root cause
       3. write an incident report
       4. run the tests
       5. open a PR if a code or contract fix applies, otherwise open an Issue
  -> A human reviews and merges
```

## Root-cause classes

| Class | Example | Agent output |
|---|---|---|
| Source defect | Vendor sends empty customer IDs or text in `amount` | GitHub Issue with the report and a recommendation to contact the vendor. No PR, because there is no code to fix. |
| Our code regression | A commit broke date normalization in `transform.py` | PR that fixes the transform and adds a regression test, with the guilty commit named in the report |
| Contract out of date | The vendor legitimately added a new valid status | PR proposing the contract change, labeled for the contract owner to approve |
| Schema drift | A column was renamed upstream | Issue with the report, plus a PR with a mapping fix if the rename is unambiguous |

## Guardrails

- The agent never merges. A human always reviews.
- Contract edits are only accepted when the root cause is contract_out_of_date, and those PRs carry the `contract-change` label.
- A PR opens only if the tests pass. Otherwise the output is an Issue.
- Values in columns marked `sensitive: true` are masked before anything is sent to the LLM or written to a report.
- Every incident has a step limit (14) and a token budget (100,000 tokens). The agent is warned when two steps remain.
- Tools are read-only: run summary, violations, file reads, code search, and git history. Reads are limited to `src/`, `tests/`, `contracts/`, `docs/`, `.github/` and two top-level files.
- Least privilege: the agent job gets only the GitHub permissions it needs.
- Audit trail per incident: run artifacts, quarantine rows, report, PR or Issue.

## Demo scenarios

1. **Source defect.** Run the defective sample (`samples/orders_2026-10-02.csv`). Seven rows break rules, 33 load, 7 are quarantined, and the agent opens an Issue.
2. **Code regression.** Commit a "harmless" change that breaks slash-date normalization in `transform.py`, then run a batch that uses slash dates. The batch fails the date rule, the agent finds the commit and opens a PR with a fix and a test.
