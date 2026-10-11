# Model evaluation

How the investigation agent was measured, what was found, and what the numbers do not show.

## Method

`contractsentinel eval` runs the agent on four scenarios whose true answer is known. Each scenario is a small git repository plus a broken batch file.

| Scenario | What is true | A correct result |
|---|---|---|
| `source_defect` | The vendor sent seven bad rows. The code is fine. | Root cause `source_defect`, no fix proposed |
| `code_regression` | A refactor removed slash-date handling, so a valid batch fails | Root cause `code_regression`, the guilty commit named, a fix that passes the tests and makes the batch pass |
| `contract_out_of_date` | A vendor notice in `docs/` announces a new region code, `MEA` | Root cause `contract_out_of_date`, a contract edit plus the test that pins the old list, tests pass, batch passes |
| `schema_drift` | The vendor renamed `customer_id` to `cust_id` | Root cause `schema_drift`, and the contract is not loosened |

Scoring is done by code, not by another model. A trial passes only if every check passes: a conclusion was reached, the root cause is right, the suspect commit is right (regression only), no fix is proposed when none is warranted, and for fixes the edit is valid, the scenario's tests pass, and the broken batch then passes the contract. Fixes are applied to a throwaway copy, and the oracle runs the scenario's own code, not the installed package.

Run it with `contractsentinel eval --models gpt-6-luna --trials 5`, or with the `Model eval` workflow.

## Results

All runs used the same four scenarios. Costs are estimates from the token counts and OpenAI's posted prices.

**Luna vs Sol (3 trials per scenario, two runs, before any tuning)**

| Model | Passed | Right root cause | Avg cost per incident |
|---|---|---|---|
| `gpt-6-luna` | 9/12 and 8/12 | 11/12 and 11/12 | about $0.004 |
| `gpt-6.1-sol` | 9/12 and 9/12 | 12/12 and 12/12 | about $0.057 |

The two models passed about the same number of trials, and Sol cost roughly 14 times more per incident. Both failed the `contract_out_of_date` scenario on every trial: the fix repaired the batch but broke a test that pinned the old list of regions. The agent could read files but could not search for where a value is used.

**Luna, after changes**

| Change | Trials | Passed | `contract_out_of_date` | `code_regression` |
|---|---|---|---|---|
| Baseline (earlier runs) | 24 | 17 | 0/6 | 6/6 |
| Added a `search_code` tool, a prompt hint to update tests that pin old values, and a "few steps remain" message | 20 | 16 | 4/5 | 2/5 |
| Softer step message, 14 step budget | 32 | 30 | 6/8 | 8/8 |

The `search_code` tool fixed the out-of-date contract scenario. The dip in `code_regression` in the middle row recovered in the last run. The step message was sent in only 1 of 32 trials in the last run, so it probably was not the cause of the dip. That row may simply be noise from five trials. These two changes were made together, so their separate effects are not isolated.

## Limits

- The scenarios and the scoring were written by the project author. They show the agent behaves sensibly on cases the author thought of, not on arbitrary incidents.
- Sample sizes are small (3 to 8 trials per scenario), so differences of one or two trials are not meaningful.
- Several things changed between runs, and the agent is not fully deterministic.
- Prices and model IDs came from OpenAI's pricing page and third-party sources, and may change.
- Only OpenAI models were tested. The provider interface supports others, but they have not been measured.
