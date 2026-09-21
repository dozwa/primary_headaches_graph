#!/usr/bin/env python
"""Poster Fig. 3: evaluate the example case against selected diagnoses and print every
node verdict of their rule trees (Kleene three-valued logic: TRUE / FALSE / UNKNOWN).

    python scripts/fig3_case.py --graph graph/…_graph.json --case gold/fig3_case.json \
        --codes 1.1,2.2 --json results/fig3_case.json

Standard library only; the reasoner is the vendored src/ichd3 package.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ichd3.engine.reasoner import EvalPolicy, Reasoner, RuleGraph               # noqa: E402
from ichd3.engine.patient_model import meta_to_case, normalize_duration_nodes  # noqa: E402


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


def hull(g: RuleGraph, code: str) -> list[str]:
    """Rule tree of one diagnosis in evaluation order: entity -> SATISFIED_BY -> operator -> OPERAND/GROUNDS_ON."""
    order: list[str] = []
    seen: set[str] = set()

    def walk(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        order.append(n)
        if n in g.satisfied_by:
            walk(g.satisfied_by[n])
        for m in g.operands.get(n, []):
            if "Diagnosis" not in g.labels.get(m, set()):     # cross-references to other diagnoses stay leaves
                walk(m)
        for (eid, _label, _neg) in g.grounds.get(n, []):
            if eid not in seen:
                seen.add(eid)
                order.append(eid)

    walk(code)
    return order


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph", required=True)
    ap.add_argument("--case", required=True, help="patient model JSON (meta format)")
    ap.add_argument("--codes", default="1.1,2.2")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    g = RuleGraph.from_dict(json.loads(Path(a.graph).read_text(encoding="utf-8")))
    normalize_duration_nodes(g)
    meta = json.loads(Path(a.case).read_text(encoding="utf-8"))
    km = Reasoner(g, EvalPolicy()).evaluate_case(build_case(meta))  # open world, as on the poster
    out = {"case": meta.get("vignette_id"), "policy": "open-world (EvalPolicy())", "diagnoses": {}}
    for code in a.codes.split(","):
        code = code.strip()
        nodes = {}
        for n in hull(g, code):
            v = str(km.get(n)).upper() if km.get(n) is not None else "UNKNOWN"
            label = next(iter(g.labels.get(n, {"Entity"})))
            entry = {"label": label, "verdict": v}
            p = g.props.get(n, {})
            if p.get("kind"):
                entry["kind"] = p["kind"]
            if p.get("name") and label != "Logic":
                entry["name"] = p["name"]
            nodes[n] = entry
        out["diagnoses"][code] = {"verdict": str(km.get(code)).upper(), "nodes": nodes}
        print(f"{code} {g.props.get(code, {}).get('name', '')}: {out['diagnoses'][code]['verdict']}")
        for n, e in nodes.items():
            if e["label"] in ("Criterion", "SubCriterion"):
                print(f"   {e['verdict']:<8} {n}  {e.get('name', '')}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
