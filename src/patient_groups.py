"""Keep original PID strings separate from the user-approved grouping assumption."""

import re


def numeric_pid(pid):
    match = re.fullmatch(r"[A-Za-z]*([0-9]+)[A-Za-z]*", pid)
    if match is None:
        raise ValueError(f"PID must contain one numeric component: {pid!r}")
    # Conservative grouping assumption, not verified identity; leading zeros are ignored.
    return str(int(match.group(1)))


def patient_group(row):
    return row.get("patient_group_id", row["patient_id"])
