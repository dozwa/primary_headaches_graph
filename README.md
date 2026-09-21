# ICHD-3 primary headache graph, Kleene reasoner and Gold-2 cases

Reproduction package for the poster **"An ICHD-3 Knowledge Graph for Structured Classification of
Headache Disorders: Development and Design — Primary headache disorders as machine-checkable rules"**
(Dorian Zwanzig, GMDS 2026; `poster/gmds_conf_poster_v3.pdf`).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22875899.svg)](https://doi.org/10.5281/zenodo.22875899)
Source: https://github.com/dozwa/primary_headaches_graph · Archive: https://doi.org/10.5281/zenodo.22875899 (concept DOI, resolves to the latest version)

The package is deliberately minimal: it holds what is needed to **build the graph** (JSON bundle →
Neo4j) and to **reproduce the poster's numbers** (reasoner + cases + checker), nothing else.

| part | what it is | where |
|---|---|---|
| Rule graph | ICHD-3 chapters 1–4 (77 diagnoses, 55 leaf codes) as typed logic trees over a closed vocabulary: the JSON bundle the reasoner reads, plus criterion-text and vocabulary side-cars | `graph/` |
| Neo4j graph | the same graph as an idempotent Cypher script (1 348 nodes, 1 851 relationships), the generator that derives it from the bundle, and a docker-compose file to load it | `neo4j/`, `scripts/export_bundle_to_cypher.py` |
| Kleene reasoner | deterministic three-valued evaluation (TRUE / FALSE / UNKNOWN) of every diagnosis tree: the rule-graph loader, the evaluator and the patient-model adapter, nothing else | `src/ichd3/` |
| Gold-2 cases | the 2 616 graph-derived cases of the rule test: 788 positive, 298 refuted, 1 530 near-miss; the Fig. 3 example case | `gold/` |
| Scripts | graph profile, rule test, Fig. 3 case, Cypher generator, a one-shot `reproduce.sh`, and a checker that asserts every number on the poster | `scripts/`, `reproduce.sh` |
| Reference results | the JSON outputs produced by this package (diffed against your run) and the result of the external validation stages 0–2 | `expected/` |

Not included: the poster's external check against independently coded cases (third-party data),
the real-data experiments, the figure rendering, and the annotation tool that authored the graph.

## Quick start

Requirements: Python ≥ 3.11. **No third-party packages** are needed for the core reproduction.

```bash
git clone https://github.com/dozwa/primary_headaches_graph.git && cd primary_headaches_graph
./reproduce.sh                                       # ≈ 3 min on a laptop
```

The last step prints a table of every poster number with observed vs. expected value and exits
with status 0 when all of them reproduce and the JSON results are identical to `expected/`.
Outputs land in `results/` (override with `OUT=…`).

Optional:

```bash
./reproduce.sh --cypher                              # regenerate neo4j/neo4j_layered.cypher from the bundle, compare and count
./reproduce.sh --neo4j                               # docker compose: load the graph into Neo4j 5, print counts
```

## What reproduces what

| step | poster number | command (see `reproduce.sh`) | output |
|---|---|---|---|
| 1 | 77 diagnoses (55 leaves; 27 / 12 / 17 / 21 per chapter), 600 logic nodes, 11 operator kinds, Fig. 1 operator histogram, node and edge counts of Fig. 2's legend (Diagnosis 77 · Criterion 166 · SubCriterion 357 · Logic 600 · Symptom 31 · Attribute 113 · Disorder 4; SATISFIED_BY 600 · OPERAND 584 · GROUNDS_ON 658, 21 negated · IS_A 2 · MUTUALLY_EXCLUSIVE 7 pairs) | `scripts/graph_profile.py` | `results/graph_profile.json` |
| 2 | rule test: 788 / 788 positive cases TRUE, 298 / 298 refuted cases FALSE, 1 530 / 1 530 near-miss cases not TRUE (open and closed world); 0 target-verdict errors on 2 616 cases × 77 diagnoses; 833 source nodes, 0 back edges (DAG check); refuted cases land on the probable form in all six tested pairs; co-TRUE diagnoses explained (parents, complications, 3.1 ↔ 3.2, 4.6.3) | `scripts/rule_test.py` | `results/rule_test.json` |
| 3 | Fig. 3: the example case is TRUE for 1.1 and FALSE for 2.2, decided by criterion A; every node verdict of both trees | `scripts/fig3_case.py` | `results/fig3_case.json` |
| 4 | staged validation: 77 / 77 (stage 0 + 1), 98 coverage warnings, 70 / 77 satisfiable (stage 2) — **not re-run here**, see below | shipped result | `expected/validation_stages_ch1-4.json` |
| 5 | Fig. 2: 1 348 nodes, 1 851 edges | computed from the bundle by `check_numbers.py`; `--cypher` re-derives the Cypher and re-counts; `--neo4j` counts in a live database | `neo4j/expected_counts.json` |

Reading the numbers correctly:

- **Near-miss cases** count as "not accepted" when the target verdict is not TRUE; under open
  world all 1 530 are UNKNOWN (one feature short = open, not refuted).
- **Fig. 1, chapter 3, "PERIOD_PATTERN 11"**: the per-chapter profile of reachable nodes gives 10
  for chapter 3; the eleventh PERIOD_PATTERN node sits in chapter 1 (1.6.x). Eleven is the correct
  sub-catalogue total. `check_numbers.py` asserts 10 (chapter 3) and 11 (total).
- Operator kinds: the export holds 11 (`GROUND, AND, OR, TEMPORAL, QUANTIFIER, KOfN, NOT,
  PERIOD_PATTERN, CAUSAL_TEMPORAL, CAUSAL_COUPLING, REPORTED`); "CAUSAL 23" in Fig. 1 is
  CAUSAL_COUPLING 18 + CAUSAL_TEMPORAL 5.
- The poster's independent check against externally coded cases (96 % closed-world / 34 %
  open-world) is not part of this package.

## The graph bundle

`graph/ichd3_decomposition_v6_primary_graph.json` is the only file the reasoner loads
(`RuleGraph.from_dict` in `src/ichd3/engine/reasoner.py`). Keys:

| key | content |
|---|---|
| `kind` | node id → operator kind for `:Logic` nodes, `null` for Diagnosis / Criterion / SubCriterion |
| `labels` | node id → labels (`Diagnosis`, `Criterion`, `SubCriterion`, `Logic`) |
| `props` | node id → properties (`name`, `label`; for operators e.g. `min`, `max`, `unit`, `k`, `n`, `scope`, `windowKind`, `windowPeriod`, `rel`, `comparator`, …) |
| `satisfied_by` | Diagnosis / Criterion / SubCriterion id → the operator node that decides it (SATISFIED_BY) |
| `operands` | operator id → list of operand ids (OPERAND) |
| `operand_props` | `"src|dst"` → edge properties of cross-references (`nodes`, `missing`) |
| `grounds` | operator id → list of `[entity id, entity label, negated]` (GROUNDS_ON) |
| `taxonomy`, `exhaustive`, `exclusive`, `epistemic` | ontology axioms: IS_A children per parent, exhaustive parents, MUTUALLY_EXCLUSIVE neighbours, open/closed-world flag per entity |

Ids are ICHD-3 paths: diagnosis `1.1`, criterion `1.1/C`, sub-criterion `1.1/C/1`, operator
`1.1/n3`. Durations are stored with their unit and normalized to minutes at load time
(`normalize_duration_nodes`); frequencies are rates per day / month / year and are **not**
converted between periods (a lifetime count never becomes a monthly rate).

`*_criteria.json` (criterion text per id) and `*_vocab.json` (entity lists + surface forms) are
side-cars: the Cypher generator writes them into the Neo4j graph (`text_en`, `surface_en`); the
reasoner does not need them.

### Gold-2 case format

One JSON file per case (`gold/gold2_decomp_v6primary/<tier>/`). `meta` is the patient model the
reasoner consumes:

```json
{"meta": {"vignette_id": "…", "language": "en",
          "findings":  [{"symptom": "headache", "attribute": "unilateral_location", "status": "present", "evidence": "…"}],
          "durations": [{"symptom": "attack", "status": "present", "min_minutes": 360, "max_minutes": 600}],
          "counts":    [{"symptom": "attack", "status": "present", "value": 10, "window": "lifetime"},
                        {"symptom": "headache_day", "status": "present", "value": 10, "window": "per_period", "per": "year"}],
          "temporal_relations": [], "causal_relations": [], "disorders": [], "measurements": []},
 "target_code": "1.1", "graph_fingerprint": "3e4abc821591", …}
```

Tiers: `pos_native` (788; up to twelve variants per diagnosis, expected TRUE), `contra` (298;
one criterion falsified, `falsified_criterion`, expected FALSE), `nearmiss` (1 530; one feature
dropped from a positive case, `removed_feature` / `source_case`, expected not TRUE). Files starting
with `_` are not cases. `gold/fig3_case.json` is the poster's example case. Provenance:
`gold/README.md`.

## Neo4j

`neo4j/neo4j_layered.cypher` is generated from the bundle by `scripts/export_bundle_to_cypher.py`
and writes exactly the nodes, relationships and properties that the pipeline's exporter reads
back: labels `:Ontology:{Diagnosis|Criterion|SubCriterion|Symptom|Attribute|Disorder}` and
`:Logic`, relationships `SATISFIED_BY`, `OPERAND`, `GROUNDS_ON` (`negated` flag), `IS_A`,
`MUTUALLY_EXCLUSIVE`; criterion text in `text_en`, surface forms in `surface_en`, epistemic flags
on entities. Load it with

```bash
cd neo4j && ./load.sh        # docker compose up, cypher-shell < neo4j_layered.cypher, counts
```

or pipe the file into any Neo4j 5 `cypher-shell`. `verify_counts.cypher` prints the label and
relationship histograms; `expected_counts.json` holds the expected values (1 348 / 1 851).

Two things the bundle does not carry and the Cypher therefore does not either: display names of
entities (derived from the id, `unilateral_location → "unilateral location"`) and the term
statistics (`df`, `idf`) of the annotation-tool export. Neither is read by the reasoner. Verified on
21 Sep 2026: load → re-export with the pipeline exporter → semantic comparison reports zero
differences in props, operands, SATISFIED_BY, grounds and labels; 523 / 523 criterion texts and all
entities and surface forms identical. The vocabulary side-car lists 8 temporal relations inherited
from the full graph, the primary graph uses 6 (`before`, `overlaps` occur only in chapters 5–14).

## Validation stages 0–2 (external)

The staged validation of the criterion documents (stage 0 JSON Schema, stage 1 deterministic
invariants, stage 2 satisfiability with clingo, HEAD-ASP predicate schema) runs inside the
annotation tool that authored the graph, not in this package. Its result for the 77 documents is
shipped as `expected/validation_stages_ch1-4.json` (`status`, `errors`, `warnings`, rule ids per
document): 77 / 77 pass stages 0 and 1, 98 `1:coverage` warnings (text spans left unanchored),
7 documents are UNSAT in stage 2 (`2:asp.unsat`: the probable forms 1.5, 1.5.1, 1.5.2, 2.4.1,
2.4.2, 2.4.3 and 3.5 — the unsat core is always clause B, the negated cross-reference to the base
diagnosis, which the ASP encoding treats as default negation; the Kleene evaluation and the generated
cases confirm these forms satisfiable). Stage 3 — behaviour under cases — is what `reproduce.sh`
re-runs in full.

## Provenance chain

```
ichd3_mainpart_v6            annotation-tool project (276 diagnoses, criterion documents)
   │  export → Neo4j 5 → bundle export (ichd3.runner.export_graph), reasoner fingerprint e562b5f18e64
   ▼
decomp_en_v6_primary         chapters 1–4 cut from the full graph (closure-checked: no operand leaves the
   │                         sub-catalogue); this package's graph/, reasoner fingerprint 3e4abc821591
   │  gold generator (18 Sep 2026; 12 variants × 80 seeds, 2 near-miss per case, 6 contra per diagnosis)
   ▼
gold2_decomp_v6primary       gold/ (pos_native, nearmiss, contra)
```

`MANIFEST.json` records the git commit of the source repository, the build time and a SHA-256
per file. The fingerprint `3e4abc821591` is stored in every gold case; the reasoner recomputes it
from the loaded bundle.

## Known limitations

- Cases and rules come from the same source: the rule test checks the translation of the text
  into logic, not the medicine.
- Open world has a price: criteria phrased as absences stay UNKNOWN unless a note states the
  absence; the closed-world policy trades that safety for recall.
- Two modelling findings remain: 3.1 and 3.2 overlap (ICHD-3 separates them only through
  criterion E, a policy placeholder, not a node), and 4.6.3 lacks the exclusion of 4.6.1 / 4.6.2.
- Seven probable forms are UNSAT in the clingo stage (see above); the reasoner evaluates them
  correctly under cases because it treats the negated parent as a cross-reference with slack.

## Package layout

```
README.md  LICENSE-CODE  LICENSE-DATA  NOTICE-ICHD3.md  CITATION.cff  .zenodo.json  MANIFEST.json  reproduce.sh
src/ichd3/          vendored Kleene core: engine/reasoner.py (graph loader + evaluator), engine/patient_model.py
                    (patient-model adapter, unit normalization), node_synonyms/i18n/paths (adapter helpers)
graph/              bundle: graph, criteria, vocab
gold/               gold2_decomp_v6primary/{pos_native,contra,nearmiss}/*.json, fig3_case.json, README.md
scripts/            graph_profile, rule_test, fig3_case, check_numbers, export_bundle_to_cypher
neo4j/              neo4j_layered.cypher, docker-compose.yml, load.sh, verify_counts.cypher, expected_counts.json
expected/           reference JSON produced by this package + validation_stages_ch1-4.json
results/            written by reproduce.sh (JSON results + check_numbers.md)
poster/             gmds_conf_poster_v3.pdf
```

## Licenses and citation

Code (`src/`, `scripts/`, `reproduce.sh`): MIT — `LICENSE-CODE`.
Data (`graph/`, `gold/`, `neo4j/*.cypher`, `expected/`): the terms of the International Headache
Society for ICHD-3 — free reproduction for scientific, educational or clinical use with attribution;
any commercial use requires the Society's permission via Sage Publications Ltd
(https://ichd-3.org/copyright/) — `LICENSE-DATA`, `NOTICE-ICHD3.md`.

Please cite the poster and this package as given in `CITATION.cff`:

> Zwanzig D. ICHD-3 primary headache knowledge graph, Kleene reasoner and Gold-2 cases — reproduction
> package for the GMDS 2026 poster. Zenodo, 2026. https://doi.org/10.5281/zenodo.22875899

Contact: Dorian Zwanzig, HNEE · dorian.zwanzig@hnee.de · ORCID 0009-0000-0990-3570
