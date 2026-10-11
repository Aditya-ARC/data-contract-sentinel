"""Masking for values that cross the trust boundary into an LLM prompt.

Columns marked ``sensitive: true`` in the contract keep their shape (so format
problems stay diagnosable) but lose their content: letters become A or a and
digits become 9. Other characters are kept.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from contractsentinel.contract import Contract


def mask_value(value: str) -> str:
    out = []
    for ch in value:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        else:
            out.append(ch)
    return "".join(out)


def sensitive_columns(contract: Contract) -> set[str]:
    return {f.name for f in contract.fields if f.sensitive}


def mask_violations(violations_doc: dict[str, Any], contract: Contract) -> dict[str, Any]:
    """Return a copy of a violations document with sensitive sample values masked."""
    doc = deepcopy(violations_doc)
    sensitive = sensitive_columns(contract)
    for violation in doc.get("violations", []):
        if violation.get("column") in sensitive:
            for sample in violation.get("samples", []):
                for key in ("value", "raw_value"):
                    if key in sample:
                        sample[key] = mask_value(str(sample[key]))
    return doc
