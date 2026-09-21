#!/usr/bin/env python
"""Rule test of the poster: every Gold-2 case evaluated with the Kleene reasoner against all 77
diagnoses, under the open-world and the closed-world policy.

Reports per tier and policy the verdict of the target diagnosis (TRUE / FALSE / UNKNOWN in %),
the co-TRUE diagnoses of every case (hierarchical parents, descendants, other), the pairs behind
"other" (probable forms, complications), and the DAG invariant of the rule graph
(source nodes, back edges) that makes the single memoized pass sound.

    python scripts/rule_test.py --graph graph/…_graph.json --set gold/gold2_decomp_v6primary --json results/rule_test.json

Standard library only; the reasoner is the vendored src/ichd3 package.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ichd3.engine.reasoner import EvalPolicy, Reasoner, RuleGraph      # noqa: E402
from ichd3.engine.patient_model import meta_to_case, normalize_duration_nodes  # noqa: E402

POLICIES = {"open": EvalPolicy(),
            "closed": EvalPolicy(differential="true", external_diagnosis="true", epistemic="graph")}
TIERS = ("pos_native", "contra", "nearmiss")
_VID_RE = re.compile(r"_([A-Z]?\d+(?:-\d+)*)_\d+$")


def build_case(meta: dict):
    """Patient model -> Case. REPORTED nodes (free-text findings a note reports, e.g. an aura lasting
    more than a week in 1.4.4) are carried in `meta["reported"]` and set on the case's REPORTED axis."""
    case = meta_to_case(meta, lang="en")
    for x in meta.get("reported", []) or []:
        if isinstance(x, dict) and isinstance(x.get("node"), str):
            if x.get("status") == "present":
                case.reported_present.add(x["node"])
            elif x.get("status") == "absent":
                case.reported_absent.add(x["node"])
    return case


def target_code(doc: dict, stem: str) -> str | None:
    """Target diagnosis of a case: explicit `target`, else the code encoded in the vignette id."""
    if isinstance(doc.get("target"), str):
        return doc["target"]
    vid = doc.get("meta", doc).get("vignette_id") or stem
    m = _VID_RE.search(vid)
    return m.group(1).replace("-", ".") if m else None


def dag_check(raw: dict) -> dict:
    """Source nodes and back edges over SATISFIED_BY + OPERAND (DFS three-colouring)."""
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in raw["satisfied_by"].items():
        adj[a].append(b)
    for a, lst in raw["operands"].items():
        adj[a] += lst
    sys.setrecursionlimit(100000)
    colour: dict[str, int] = {}
    back = 0

    def dfs(n: str) -> None:
        nonlocal back
        colour[n] = 1
        for m in adj.get(n, []):
            c = colour.get(m, 0)
            if c == 1:
                back += 1
            elif c == 0:
                dfs(m)
        colour[n] = 2

    for n in list(adj):
        if colour.get(n, 0) == 0:
            dfs(n)
    return {"source_nodes": len(adj), "back_edges": back}


def related(target: str, other: str) -> str:
    if other.startswith(target + "."):
        return "desc"
    if target.startswith(other + "."):
        return "anc"
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", required=True)
    ap.add_argument("--set", required=True, help="folder with the tier sub-folders")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--limit", type=int, help="cases per tier (smoke test)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    raw = json.loads(Path(a.graph).read_text(encoding="utf-8"))
    g = RuleGraph.from_dict(raw)
    normalize_duration_nodes(g)                       # h/min -> minutes; mandatory before evaluation
    reasoners = {name: Reasoner(g, pol) for name, pol in POLICIES.items()}
    diags = sorted(n for n, l in g.labels.items() if "Diagnosis" in l)
    out = {"graph": a.graph, "dag": dag_check(raw), "verdicts": {}, "crosstalk": {}}
    print(f"DAG check: {out['dag']['source_nodes']} source nodes, {out['dag']['back_edges']} back edges")

    for tier in a.tiers.split(","):
        files = sorted(p for p in (Path(a.set) / tier).glob("*.json") if not p.name.startswith("_"))
        if a.limit:
            files = files[:a.limit]
        cases = []
        for f in files:
            doc = json.loads(f.read_text(encoding="utf-8"))
            code = target_code(doc, f.stem)
            if code and code in g.labels:
                cases.append((code, build_case(doc.get("meta", doc))))
        for pol, rsn in reasoners.items():
            counts = Counter(str(rsn.evaluate_case(case).get(code)).upper() for code, case in cases)
            n = len(cases)
            out["verdicts"][f"{tier}/{pol}"] = {"n": n, "verdict": {k: round(100 * v / n, 4) for k, v in sorted(counts.items())}}
            print(f"{tier:<12} {pol:<7} n={n:<5} " + "  ".join(f"{k} {100 * v / n:5.1f}%" for k, v in sorted(counts.items())))
        # crosstalk under the open-world policy (the poster's reading)
        rsn = reasoners["open"]
        tv, co, co_cases, other_pairs, n_true = Counter(), Counter(), Counter(), Counter(), Counter()
        for code, case in cases:
            km = rsn.evaluate_case(case)
            tv[str(km.get(code)).upper()] += 1
            trues = [d for d in diags if d != code and str(km.get(d)).upper() == "TRUE"]
            n_true[len(trues)] += 1
            kinds = set()
            for d in trues:
                r = related(code, d)
                co[r] += 1
                kinds.add(r)
                if r == "other":
                    other_pairs[(code, d)] += 1
            for k in kinds:
                co_cases[k] += 1
            if not trues:
                co_cases["none"] += 1
        out["crosstalk"][tier] = {
            "n": len(cases), "target_verdict": dict(tv), "co_true_pairs": dict(co),
            "cases_with_co_true": dict(co_cases),
            "n_other_true_per_case": {str(k): v for k, v in sorted(n_true.items())},
            "other_pairs_top": [[list(p), c] for p, c in other_pairs.most_common(15)]}
        print(f"{tier:<12} crosstalk: target {dict(tv)}, co-TRUE {dict(co)}, "
              f"top other pairs {other_pairs.most_common(4)}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
