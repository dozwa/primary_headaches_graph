#!/usr/bin/env python
"""Assert the numbers printed on the GMDS 2026 poster (v3) against the JSON files produced by
reproduce.sh, and diff results/ against expected/ (volatile keys ignored).

    python scripts/check_numbers.py --results results --expected expected --graph graph/…_graph.json

Exit code 0 = every poster number reproduced and no structural difference; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VOLATILE = {"seconds", "path", "graph", "elapsed"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(x) for x in o]
    if isinstance(o, float):
        return round(o, 6)
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--expected", default="expected")
    ap.add_argument("--graph", required=True)
    a = ap.parse_args(argv)
    R, E = Path(a.results), Path(a.expected)

    rows: list[tuple[str, object, object]] = []

    def check(name, observed, expected, tol=0.0):
        rows.append((name, observed, expected, abs(observed - expected) <= tol if isinstance(observed, (int, float))
                     and isinstance(expected, (int, float)) and not isinstance(observed, bool) else observed == expected))

    # --- graph profile (poster header, Fig. 1, Fig. 2 legend)
    prof = load(R / "graph_profile.json")
    prof = prof[0] if isinstance(prof, list) else prof
    check("diagnoses (chapters 1-4)", prof["diagnoses"], 77)
    check("leaf diagnoses", prof["leaves"], 55)
    for ch, n in (("1", 27), ("2", 12), ("3", 17), ("4", 21)):
        check(f"chapter {ch} diagnoses", prof["chapters"][ch]["diagnoses"], n)
    check("Logic nodes", prof["labels"]["Logic"], 600)
    check("operator kinds in export", len(prof["kinds"]), 11)
    for label, n in (("Criterion", 166), ("SubCriterion", 357), ("Diagnosis", 77)):
        check(f"{label} nodes", prof["labels"][label], n)
    for label, n in (("Symptom", 31), ("Attribute", 113), ("Disorder", 4)):
        check(f"{label} entities", prof["entities"][label], n)
    check("OPERAND edges", prof["edges_operands"], 584)
    check("SATISFIED_BY edges", prof["edges_satisfied_by"], 600)
    check("GROUNDS_ON edges", prof["ground_refs"], 658)
    check("GROUNDS_ON negated", prof["negated_ground_refs"], 21)
    check("MUTUALLY_EXCLUSIVE pairs", prof["exclusive_pairs"] // 2, 7)
    fig1 = {"1": {"TEMPORAL": 79, "OR": 63, "KOfN": 29, "QUANTIFIER": 32},
            "2": {"QUANTIFIER": 33, "KOfN": 16, "NOT": 3},
            # Poster Fig. 1 prints "PERIOD_PATTERN 11" for chapter 3; per-chapter reachable count is 10
            # (one PERIOD_PATTERN node belongs to chapter 1); 11 is the sub-catalogue total.
            "3": {"OR": 126, "QUANTIFIER": 40, "TEMPORAL": 22, "PERIOD_PATTERN": 10},
            "4": {"TEMPORAL": 31, "QUANTIFIER": 20, "OR": 16}}
    for ch, kinds in fig1.items():
        for k, n in kinds.items():
            check(f"Fig.1 chapter {ch} {k}", prof["chapters"][ch]["kinds"].get(k, 0), n)
    k4 = prof["chapters"]["4"]["kinds"]
    check("Fig.1 chapter 4 CAUSAL (coupling+temporal)", k4.get("CAUSAL_COUPLING", 0) + k4.get("CAUSAL_TEMPORAL", 0), 23)
    check("PERIOD_PATTERN total (sub-catalogue)", prof["kinds"]["PERIOD_PATTERN"], 11)

    # --- bundle-level counts (Fig. 2: 1348 nodes / 1851 edges)
    g = load(Path(a.graph))
    entities = {t[0] for v in g["grounds"].values() for t in v}
    n_nodes = len(g["labels"]) + len(entities)
    is_a = sum(len(v) for v in g.get("taxonomy", {}).values())
    me = len({tuple(sorted((x, y))) for x, ys in g.get("exclusive", {}).items() for y in ys})
    n_edges = (len(g["satisfied_by"]) + sum(len(v) for v in g["operands"].values())
               + sum(len(v) for v in g["grounds"].values()) + is_a + me)
    check("Fig.2 nodes (rule + entity)", n_nodes, 1348)
    check("Fig.2 edges", n_edges, 1851)
    check("IS_A edges (bundle)", is_a, 2)
    check("MUTUALLY_EXCLUSIVE pairs (bundle)", me, 7)

    # --- rule test (Gold-2 on the primary graph)
    rt = load(R / "rule_test.json")
    ev = rt["verdicts"]
    check("pos_native cases", ev["pos_native/open"]["n"], 788)
    check("pos_native TRUE % (open world)", ev["pos_native/open"]["verdict"].get("TRUE", 0), 100.0)
    check("contra cases", ev["contra/open"]["n"], 298)
    check("contra FALSE % (open world)", ev["contra/open"]["verdict"].get("FALSE", 0), 100.0)
    check("nearmiss cases", ev["nearmiss/open"]["n"], 1530)
    check("nearmiss TRUE % (open world)", ev["nearmiss/open"]["verdict"].get("TRUE", 0), 0.0)
    check("pos_native TRUE % (closed world)", ev["pos_native/closed"]["verdict"].get("TRUE", 0), 100.0)
    check("contra FALSE % (closed world)", ev["contra/closed"]["verdict"].get("FALSE", 0), 100.0)

    # --- crosstalk: target verdicts, DAG invariant, probable-form fallback
    ct = rt["crosstalk"]
    ct["dag"] = rt["dag"]
    check("DAG source nodes", ct["dag"]["source_nodes"], 833)
    check("DAG back edges", ct["dag"]["back_edges"], 0)
    errors = (ct["pos_native"]["n"] - ct["pos_native"]["target_verdict"].get("TRUE", 0)
              + ct["contra"]["n"] - ct["contra"]["target_verdict"].get("FALSE", 0)
              + ct["nearmiss"]["target_verdict"].get("TRUE", 0))
    check("target verdict errors (2616 cases)", errors, 0)
    check("cases checked", ct["pos_native"]["n"] + ct["contra"]["n"] + ct["nearmiss"]["n"], 2616)
    pairs = {tuple(p): n for p, n in ct["contra"]["other_pairs_top"]}
    probable = [("1.1", "1.5.1"), ("1.2", "1.5.2"), ("2.1", "2.4.1"), ("2.2", "2.4.2"), ("2.3", "2.4.3"), ("3.4", "3.5")]
    check("probable-form fallback pairs (6 contra each)", sum(1 for p in probable if pairs.get(p) == 6), 6)

    # --- Fig. 3
    f3 = load(R / "fig3_case.json")["diagnoses"]
    check("Fig.3 1.1 verdict", f3["1.1"]["verdict"], "TRUE")
    check("Fig.3 2.2 verdict", f3["2.2"]["verdict"], "FALSE")
    check("Fig.3 2.2/A verdict", f3["2.2"]["nodes"]["2.2/A"]["verdict"], "FALSE")
    check("Fig.3 2.2/C verdict", f3["2.2"]["nodes"]["2.2/C"]["verdict"], "UNKNOWN")

    # --- validation stages (external run, shipped result)
    val = load(E / "validation_stages_ch1-4.json")
    check("validation: documents", len(val), 77)
    check("stage 0/1: documents with errors", sum(1 for x in val.values() if any(not r.startswith("2:") for r in x["error_rules"])), 0)
    check("stage 1: coverage warnings", sum(x["warnings"] for x in val.values()), 98)
    check("stage 2 (clingo): UNSAT documents", sum(1 for x in val.values() if "2:asp.unsat" in x["error_rules"]), 7)

    # --- structural diff results vs expected
    diffs = []
    for name in ("graph_profile.json", "rule_test.json", "fig3_case.json"):
        if (E / name).exists() and (R / name).exists() and strip(load(E / name)) != strip(load(R / name)):
            diffs.append(name)

    w = max(len(r[0]) for r in rows)
    print(f"| {'poster number':<{w}} | observed | expected | ok |")
    print(f"|{'-' * (w + 2)}|----------|----------|----|")
    for name, obs, exp, ok in rows:
        fo = f"{obs:.2f}" if isinstance(obs, float) else str(obs)
        fe = f"{exp:.2f}" if isinstance(exp, float) else str(exp)
        print(f"| {name:<{w}} | {fo:>8} | {fe:>8} | {'✓' if ok else '✗'} |")
    n_bad = sum(1 for r in rows if not r[3])
    print(f"\n{len(rows) - n_bad}/{len(rows)} poster numbers reproduced; "
          + ("results identical to expected/" if not diffs else "DIFFERENCES vs expected/: " + ", ".join(diffs)))
    return 1 if n_bad or diffs else 0


if __name__ == "__main__":
    sys.exit(main())
