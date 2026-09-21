"""Quantitatives Profil eines Graph-Bundles — gesamt und je Kapitel (Diagnosen, Knotenarten,
Operatoren, Grounds, Entitäten, Tiefe, Kriterien, Exklusiv-/Taxonomie-/Epistemik-Deklarationen).

    python scripts/analysis/graph_profile.py --graph <graph.json> [--graph <graph2.json>] [--md out.md]
"""
from __future__ import annotations
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path


def profile(path: str) -> dict:
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = dict(g["kind"]); labels = dict(g["labels"]); props = dict(g["props"])
    operands = dict(g["operands"]); grounds = dict(g["grounds"]); sat = defaultdict(list)
    def items(x):
        return x.items() if isinstance(x, dict) else x
    for crit, node in items(g["satisfied_by"]):
        for nd in (node if isinstance(node, list) else [node]):
            sat[crit].append(nd)
    diags = sorted(n for n, l in labels.items() if "Diagnosis" in l)
    # Erreichbarkeit je Diagnose (transitive Hülle über satisfied_by + operands)
    def hull(d: str) -> set[str]:
        seen, stack = set(), [d]
        while stack:
            n = stack.pop()
            if n in seen: continue
            seen.add(n)
            stack.extend(sat.get(n, [])); stack.extend(operands.get(n, []))
        return seen
    def depth(d: str) -> int:
        memo = {}
        def dp(n, path):
            if n in memo: return memo[n]
            if n in path: return 0
            kids = sat.get(n, []) + operands.get(n, [])
            memo[n] = 0 if not kids else 1 + max(dp(k, path | {n}) for k in kids)
            return memo[n]
        return dp(d, frozenset())
    ent_type = {}
    for n, lst in grounds.items():
        for e, t, _neg in lst: ent_type[e] = t
    per_ch = {}
    node_owner = {}
    for d in diags:
        ch = d.split(".")[0]
        h = hull(d)
        for n in h: node_owner.setdefault(n, ch)
        c = per_ch.setdefault(ch, {"diagnoses": 0, "leaves": 0, "criteria": 0, "kinds": Counter(), "ents": {}, "depth": [], "grounds_per_dx": [], "letters_per_dx": []})
        c["diagnoses"] += 1
        letters = [x for x in h if "Criterion" in labels.get(x, []) and "SubCriterion" not in labels.get(x, [])]
        c["letters_per_dx"].append(len(letters))
        c["grounds_per_dx"].append(sum(1 for x in h if kind.get(x) == "GROUND"))
        c["depth"].append(depth(d))
        for x in h:
            if kind.get(x): c["kinds"][kind[x]] += 1
        for x in h:
            for e, t, _neg in grounds.get(x, []): c["ents"][e] = t
    children = defaultdict(int)
    for d in diags:
        p = d.rsplit(".", 1)[0]
        if p in labels: children[p] += 1
    for ch, c in per_ch.items():
        c["leaves"] = sum(1 for d in diags if d.split(".")[0] == ch and children.get(d, 0) == 0)
    total_kinds = Counter(v for v in kind.values() if v)
    out = {"path": path, "nodes": len(labels), "diagnoses": len(diags), "leaves": sum(1 for d in diags if children.get(d, 0) == 0),
           "labels": dict(Counter(tuple(l)[0] for l in labels.values())), "kinds": dict(total_kinds),
           "edges_operands": sum(len(v) for v in operands.values()), "edges_satisfied_by": sum(len(v) for v in sat.values()),
           "ground_nodes": len(grounds), "ground_refs": sum(len(v) for v in grounds.values()),
           "negated_ground_refs": sum(1 for v in grounds.values() for _e, _t, neg in v if neg),
           "entities": dict(Counter(ent_type.values())),
           "shared_nodes": sum(1 for n in labels if "|" in n),
           "taxonomy": len(g.get("taxonomy", [])), "exhaustive": len(g.get("exhaustive", [])), "exclusive_pairs": sum(len(v) for _k, v in items(g.get("exclusive", []))),
           "epistemic": dict(Counter(v for _k, v in items(g.get("epistemic", [])))),
           "chapters": {}}
    for ch, c in sorted(per_ch.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 99):
        out["chapters"][ch] = {"diagnoses": c["diagnoses"], "leaves": c["leaves"],
            "kinds": dict(c["kinds"]), "entities": dict(Counter(c["ents"].values())),
            "depth_median": statistics.median(c["depth"]), "depth_max": max(c["depth"]),
            "letters_per_dx_median": statistics.median(c["letters_per_dx"]),
            "grounds_per_dx_median": statistics.median(c["grounds_per_dx"]), "grounds_per_dx_max": max(c["grounds_per_dx"])}
    return out


def md(profiles: list[dict]) -> str:
    L = ["| Größe | " + " | ".join(Path(p["path"]).parent.name for p in profiles) + " |", "|---|" + "---:|" * len(profiles)]
    def row(name, f): L.append(f"| {name} | " + " | ".join(str(f(p)) for p in profiles) + " |")
    row("Knoten", lambda p: p["nodes"]); row("Diagnosen (davon Blätter)", lambda p: f"{p['diagnoses']} ({p['leaves']})")
    for lab in ("Criterion", "SubCriterion", "Logic"): row(f"Label {lab}", lambda p, lab=lab: p["labels"].get(lab, 0))
    for k in ("GROUND", "AND", "OR", "KOfN", "NOT", "TEMPORAL", "QUANTIFIER", "CAUSAL_COUPLING", "CAUSAL_TEMPORAL", "PERIOD_PATTERN", "REPORTED"):
        row(f"Operator {k}", lambda p, k=k: p["kinds"].get(k, 0))
    row("Kanten satisfied_by / operands", lambda p: f"{p['edges_satisfied_by']} / {p['edges_operands']}")
    row("Ground-Referenzen (davon negiert)", lambda p: f"{p['ground_refs']} ({p['negated_ground_refs']})")
    for t in ("Symptom", "Attribute", "Disorder"): row(f"Entitäten {t}", lambda p, t=t: p["entities"].get(t, 0))
    row("Geteilte Knoten (a|b)", lambda p: p["shared_nodes"])
    row("Taxonomie / exhaustiv / exklusive Paare", lambda p: f"{p['taxonomy']} / {p['exhaustive']} / {p['exclusive_pairs']}")
    row("Epistemik closed_world", lambda p: p["epistemic"].get("closed_world", 0))
    L.append("")
    for p in profiles:
        L.append(f"**Je Kapitel — {Path(p['path']).parent.name}**\n")
        L.append("| Kap. | Diagnosen (Blätter) | Tiefe med/max | Buchstaben/Dx med | Grounds/Dx med/max | Sym/Attr/Dis | AND | OR | KOfN | NOT | TEMPORAL | QUANTIFIER | CAUSAL |")
        L.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for ch, c in p["chapters"].items():
            k = c["kinds"]; e = c["entities"]
            L.append(f"| {ch} | {c['diagnoses']} ({c['leaves']}) | {c['depth_median']:.0f}/{c['depth_max']} | {c['letters_per_dx_median']:.0f} | "
                     f"{c['grounds_per_dx_median']:.0f}/{c['grounds_per_dx_max']} | {e.get('Symptom',0)}/{e.get('Attribute',0)}/{e.get('Disorder',0)} | "
                     f"{k.get('AND',0)} | {k.get('OR',0)} | {k.get('KOfN',0)} | {k.get('NOT',0)} | {k.get('TEMPORAL',0)} | {k.get('QUANTIFIER',0)} | {k.get('CAUSAL_COUPLING',0)+k.get('CAUSAL_TEMPORAL',0)} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph", action="append", required=True)
    ap.add_argument("--md"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    ps = [profile(p) for p in a.graph]
    text = md(ps); print(text)
    if a.md: Path(a.md).write_text(text, encoding="utf-8")
    if a.json: Path(a.json).write_text(json.dumps(ps, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
