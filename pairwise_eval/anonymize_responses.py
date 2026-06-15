"""Produce a shareable, anonymized copy of the human pairwise responses.

The raw export carries rater names and a browser user-agent string. This maps
each rater to a stable opaque id (R01, R02, ... in order of first appearance) and
drops the user-agent column, keeping everything the kappa analysis needs
(pair_id, q1, q2) plus timing. The raw file stays local; only the output is
meant to be committed.

Usage:
    uv run python anonymize_responses.py [in.csv] [out.csv]
Defaults: human_responses.csv -> human_responses_anon.csv
"""
import csv
import sys

DROP_COLUMNS = {"user_agent"}

src = sys.argv[1] if len(sys.argv) > 1 else "human_responses.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "human_responses_anon.csv"

with open(src, newline="") as f:
    reader = csv.DictReader(f)
    fields = [c for c in reader.fieldnames if c not in DROP_COLUMNS]
    rows = list(reader)

ids = {}
for r in rows:
    if r["rater"] not in ids:
        ids[r["rater"]] = f"R{len(ids) + 1:02d}"

with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        r["rater"] = ids[r["rater"]]
        w.writerow(r)

print(f"{len(ids)} raters -> {dst} ({len(rows)} rows); dropped {sorted(DROP_COLUMNS)}")
