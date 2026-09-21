#!/usr/bin/env python
"""Bundle-JSON -> Cypher: den Offline-Graphen wieder in eine Neo4j-Datenbank laden.

Die Pipeline exportiert Neo4j -> Bundle (``ichd3.runner.export_graph``); die Gegenrichtung
gab es nicht. Fuer das Publikationspaket muss ein Leser den Graphen ohne das
Annotation-Tool in Neo4j sehen koennen. Dieses Skript erzeugt aus den drei Bundle-Dateien
(graph / vocab / criteria) genau die Knoten, Kanten und Eigenschaften, die
``RuleGraph.from_neo4j``, ``Vocabulary.from_neo4j`` und ``export_criteria`` zurücklesen —
so dass ``python -m ichd3.runner.export_graph`` gegen die geladene Datenbank das Bundle
semantisch reproduziert (``scripts/compare_bundles.py``).

Zeilengrammatik wie der Export des Annotation-Tools (ein MERGE-Statement je Zeile,
idempotent), damit ``scripts/verify_export_vs_bundle.py`` unveraendert prueft.

Was das Bundle NICHT traegt und deshalb aus der id abgeleitet wird: Entitaetsnamen
(``name`` = id mit Leerzeichen statt Unterstrich). ``df``/``idf`` des Tool-Exports
entfallen (vom Reasoner nie gelesen).

    python scripts/export_bundle_to_cypher.py \
        --graph assets/graphs/decomp_en_v6_primary/ichd3_decomposition_v6_primary_graph.json \
        --vocab assets/graphs/decomp_en_v6_primary/ichd3_decomposition_v6_primary_vocab.json \
        --criteria assets/graphs/decomp_en_v6_primary/ichd3_decomposition_v6_primary_criteria.json \
        --out publication/gmds2026/neo4j/neo4j_layered.cypher --check
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ENTITY_LABELS = {"symptoms": "Symptom", "attributes": "Attribute", "disorders": "Disorder"}
RULE_LABELS = ("Diagnosis", "Criterion", "SubCriterion")
CONSTRAINT_LABELS = ("Diagnosis", "Criterion", "SubCriterion", "Symptom", "Attribute", "Disorder", "Logic")


def q(s: str) -> str:
    """Cypher-String-Literal in einfachen Anfuehrungszeichen."""
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def lit(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return q(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    raise TypeError(f"nicht serialisierbar: {type(v).__name__} ({v!r})")


def props_clause(var: str, props: dict, *, skip=("id",)) -> str:
    items = [(k, v) for k, v in props.items() if k not in skip and v is not None]
    return ", ".join(f"{var}.{k}={lit(v)}" for k, v in items)


def code_key(code: str):
    """Sortierschluessel fuer Codes und Knoten-ids (1.1 < 1.10; 1.1/A < 1.1/n2): jedes Segment als
    (0, Zahl) oder (1, Text), damit gemischte Segmente vergleichbar bleiben."""
    return tuple((0, int(x)) if x.isdigit() else (1, x) for x in re.split(r"[./]", code) if x != "")


def fingerprint(graph: dict) -> str:
    return hashlib.sha256(json.dumps(graph, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def generate(graph: dict, vocab: dict, criteria: dict, title: str) -> tuple[list[str], dict]:
    labels = {k: set(v) for k, v in graph["labels"].items()}
    props = graph["props"]
    grounds = graph.get("grounds", {})
    taxonomy = graph.get("taxonomy", {})
    exhaustive = set(graph.get("exhaustive", []))
    exclusive = graph.get("exclusive", {})
    epistemic = graph.get("epistemic", {})
    operand_props = {tuple(k.split("|", 1)): v for k, v in graph.get("operand_props", {}).items()}

    # --- Entitaeten: Universum = Vokabular-Listen, Label aus der Liste; Grounds muessen darin liegen.
    entity_label: dict[str, str] = {}
    for key, label in ENTITY_LABELS.items():
        for eid in vocab.get(key, []):
            entity_label[eid] = label
    referenced = {t[0]: t[1] for v in grounds.values() for t in v}
    for eid, label in referenced.items():
        if entity_label.get(eid) != label:
            raise SystemExit(f"Ground-Ziel {eid!r} ({label}) nicht im Vokabular oder Label weicht ab")
    for eid in set(epistemic) | exhaustive | set(exclusive) | set(taxonomy) | {c for v in taxonomy.values() for c in v}:
        if eid not in entity_label:
            raise SystemExit(f"Axiom-Knoten {eid!r} ist keine Vokabular-Entitaet")
    surface = vocab.get("surface", {})

    lines: list[str] = []
    n_diag = sum(1 for l in labels.values() if "Diagnosis" in l)
    lines += [
        f"// {title}",
        f"// erzeugt aus dem Bundle-JSON (Graph-Fingerprint {fingerprint(graph)}) — {n_diag} Diagnose(n), "
        f"{len(entity_label)} Entitaeten, {sum(1 for l in labels.values() if 'Logic' in l)} Logikknoten",
        f"// Generator: scripts/export_bundle_to_cypher.py, {_dt.date.today().isoformat()} — idempotent (MERGE), ein Statement je Zeile",
        "// Ladeweg: cypher-shell -u neo4j -p <pw> < neo4j_layered.cypher   (Neo4j 5)",
        "",
        "// --- Eindeutigkeits-Constraints (legen zugleich den Index fuer die MATCHes unten an)",
    ]
    for label in CONSTRAINT_LABELS:
        lines.append(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE;")
    lines += ["", "// --- Ontologie-Entitaeten (Symptom / Attribute / Disorder)"]
    for eid in sorted(entity_label, key=lambda e: (entity_label[e], e)):
        label = entity_label[eid]
        p: dict = {"name": eid.replace("_", " ")}
        if eid in epistemic:
            p["epistemic"] = epistemic[eid]
        if eid in exhaustive:
            p["exhaustive"] = True
        kind = {"Symptom": "symptom", "Attribute": "attribute", "Disorder": "disorder"}[label]
        forms = surface.get(kind, {}).get(eid)
        if forms:
            p["surface_en"] = list(forms)
        lines.append(f"MERGE (n:Ontology:{label} {{id:{q(eid)}}}) SET {props_clause('n', p)};")
    lines += ["", "// --- Ontologie-Axiome (IS_A, MUTUALLY_EXCLUSIVE)"]
    for parent, children in sorted(taxonomy.items()):
        for child in children:
            lines.append(f"MATCH (a:Ontology:{entity_label[child]} {{id:{q(child)}}}), "
                         f"(b:Ontology:{entity_label[parent]} {{id:{q(parent)}}}) MERGE (a)-[r:IS_A]->(b);")
    pairs = sorted({tuple(sorted((a, b))) for a, bs in exclusive.items() for b in bs})
    for a, b in pairs:
        lines.append(f"MATCH (a:Ontology:{entity_label[a]} {{id:{q(a)}}}), "
                     f"(b:Ontology:{entity_label[b]} {{id:{q(b)}}}) MERGE (a)-[r:MUTUALLY_EXCLUSIVE]->(b);")

    # --- Regelknoten je Diagnose (Diagnosis, Criterion, SubCriterion, Logic + Kanten)
    def diag_of(nid: str) -> str:
        return nid.split("/", 1)[0]

    by_diag: dict[str, list[str]] = {}
    for nid, l in labels.items():
        by_diag.setdefault(diag_of(nid), []).append(nid)
    counts = Counter()
    for code in sorted(by_diag, key=code_key):
        nodes = sorted(by_diag[code], key=lambda n: (0 if "Diagnosis" in labels[n] else 1 if "Criterion" in labels[n]
                                                     else 2 if "SubCriterion" in labels[n] else 3, code_key(n), n))
        lines += ["", f"// ===== {code}"]
        for nid in nodes:
            l = labels[nid]
            if "Logic" in l:
                p = dict(props[nid])
                lines.append(f"MERGE (n:Logic {{id:{q(nid)}}}) SET {props_clause('n', p)};")
                counts["Logic"] += 1
            else:
                label = next(x for x in RULE_LABELS if x in l)
                p = dict(props.get(nid, {}))
                if nid in criteria:
                    p["text_en"] = list(criteria[nid]) if isinstance(criteria[nid], list) else [criteria[nid]]
                var = "d" if label == "Diagnosis" else "n"
                lines.append(f"MERGE ({var}:Ontology:{label} {{id:{q(nid)}}}) SET {props_clause(var, p)};")
                counts[label] += 1
        for nid in nodes:
            l = labels[nid]
            if "Logic" in l:
                continue
            tgt = graph["satisfied_by"].get(nid)
            if tgt:
                label = next(x for x in RULE_LABELS if x in l)
                lines.append(f"MATCH (a:Ontology:{label} {{id:{q(nid)}}}), (b:Logic {{id:{q(tgt)}}}) "
                             f"MERGE (a)-[r:SATISFIED_BY]->(b);")
                counts["SATISFIED_BY"] += 1
        for nid in nodes:
            if "Logic" not in labels[nid]:
                continue
            for dst in graph.get("operands", {}).get(nid, []):
                dl = labels[dst]
                dlab = "Logic" if "Logic" in dl else "Ontology:" + next(x for x in RULE_LABELS if x in dl)
                extra = operand_props.get((nid, dst))
                tail = f" SET {props_clause('r', extra, skip=())}" if extra else ""
                lines.append(f"MATCH (a:Logic {{id:{q(nid)}}}), (b:{dlab} {{id:{q(dst)}}}) MERGE (a)-[r:OPERAND]->(b){tail};")
                counts["OPERAND"] += 1
            for (eid, elabel, neg) in grounds.get(nid, []):
                tail = " SET r.negated=true" if neg else ""
                lines.append(f"MATCH (a:Logic {{id:{q(nid)}}}), (b:Ontology:{elabel} {{id:{q(eid)}}}) "
                             f"MERGE (a)-[r:GROUNDS_ON]->(b){tail};")
                counts["GROUNDS_ON"] += 1
                counts["GROUNDS_ON_negated"] += int(bool(neg))
    for eid, label in entity_label.items():
        counts[label] += 1
    counts["IS_A"] = sum(len(v) for v in taxonomy.values())
    counts["MUTUALLY_EXCLUSIVE"] = len(pairs)
    node_total = sum(counts[k] for k in ("Diagnosis", "Criterion", "SubCriterion", "Logic", "Symptom", "Attribute", "Disorder"))
    edge_total = sum(counts[k] for k in ("SATISFIED_BY", "OPERAND", "GROUNDS_ON", "IS_A", "MUTUALLY_EXCLUSIVE"))
    summary = {"nodes": node_total, "edges": edge_total, "labels": {k: counts[k] for k in
               ("Diagnosis", "Criterion", "SubCriterion", "Logic", "Symptom", "Attribute", "Disorder")},
               "relationships": {k: counts[k] for k in ("SATISFIED_BY", "OPERAND", "GROUNDS_ON", "IS_A", "MUTUALLY_EXCLUSIVE")},
               "grounds_on_negated": counts["GROUNDS_ON_negated"], "graph_fingerprint": fingerprint(graph)}
    return lines, summary


def check(cypher_path: Path, graph: dict, expected: dict | None) -> int:
    """Erzeugte Datei re-parsen: Statement-Zaehler je Typ gegen die Zusammenfassung / expected_counts."""
    text = cypher_path.read_text(encoding="utf-8").splitlines()
    stmts = [l for l in text if l and not l.startswith("//")]
    c = Counter()
    for l in stmts:
        m = re.match(r"MERGE \((?:n|d):(?:Ontology:)?(\w+) \{id:", l)
        if m:
            c[m.group(1)] += 1
            continue
        m = re.search(r"MERGE \(a\)-\[r:(\w+)\]->\(b\)", l)
        if m:
            c[m.group(1)] += 1
    c["negated"] = sum(1 for l in stmts if "SET r.negated=true" in l)
    nodes = sum(c[k] for k in ("Diagnosis", "Criterion", "SubCriterion", "Logic", "Symptom", "Attribute", "Disorder"))
    edges = sum(c[k] for k in ("SATISFIED_BY", "OPERAND", "GROUNDS_ON", "IS_A", "MUTUALLY_EXCLUSIVE"))
    print(f"check: {len(stmts)} Statements, nodes={nodes} edges={edges}, " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    bad = 0
    if expected:
        for k, v in expected.get("labels", {}).items():
            bad += c[k] != v
        for k, v in expected.get("relationships", {}).items():
            bad += c[k] != v
        bad += nodes != expected.get("nodes", nodes) or edges != expected.get("edges", edges)
        bad += c["negated"] != expected.get("grounds_on_negated", c["negated"])
        print("check gegen expected_counts:", "OK" if not bad else f"{bad} Abweichung(en)")
    # Jede MATCH-Kante muss auf vorher erzeugte Knoten zeigen.
    ids = {m.group(1) for l in stmts for m in [re.match(r"MERGE \((?:n|d):[\w:]+ \{id:'((?:[^'\\]|\\.)*)'\}\)", l)] if m}
    dangling = 0
    for l in stmts:
        for m in re.finditer(r"\{id:'((?:[^'\\]|\\.)*)'\}", l):
            if m.group(1) not in ids:
                dangling += 1
    print(f"check: {len(ids)} Knoten-ids, {dangling} haengende Verweise")
    return int(bool(bad or dangling))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="ICHD-3 primary headache graph (chapters 1-4) — Neo4j layered export")
    ap.add_argument("--summary-json", help="Zaehler je Label/Kantentyp als JSON schreiben (expected_counts.json)")
    ap.add_argument("--check", action="store_true", help="Datei re-parsen und Zaehler gegen --expected (oder die eigene Zusammenfassung) pruefen")
    ap.add_argument("--expected", help="expected_counts.json fuer --check")
    a = ap.parse_args(argv)
    graph = json.loads(Path(a.graph).read_text(encoding="utf-8"))
    vocab = json.loads(Path(a.vocab).read_text(encoding="utf-8"))
    criteria = json.loads(Path(a.criteria).read_text(encoding="utf-8"))
    lines, summary = generate(graph, vocab, criteria, a.title)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out} ({len(lines)} Zeilen, {out.stat().st_size} B)  nodes={summary['nodes']} edges={summary['edges']}")
    if a.summary_json:
        Path(a.summary_json).write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.check:
        expected = json.loads(Path(a.expected).read_text(encoding="utf-8")) if a.expected else summary
        return check(out, graph, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
