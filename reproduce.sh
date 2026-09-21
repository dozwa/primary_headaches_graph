#!/usr/bin/env bash
# Reproduce the numbers of the GMDS 2026 poster from this package alone (Python >= 3.11, no packages).
#   ./reproduce.sh            graph profile, rule test (2616 cases x 77 diagnoses), Fig. 3, number check  (~3 min)
#   ./reproduce.sh --cypher   + regenerate neo4j/neo4j_layered.cypher from the bundle and check its counts
#   ./reproduce.sh --neo4j    + load the graph into Neo4j 5 via docker compose and print counts
# Environment: PYTHON (default python3), OUT (default ./results)
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG"
export ICHD3_ROOT="$PKG"
PY="${PYTHON:-python3}"
OUT="${OUT:-$PKG/results}"
mkdir -p "$OUT"
G="$PKG/graph/ichd3_decomposition_v6_primary"
GRAPH="${G}_graph.json"
GOLD="$PKG/gold/gold2_decomp_v6primary"
CYPHER=0; NEO=0
for arg in "$@"; do
  case "$arg" in
    --cypher) CYPHER=1 ;;
    --neo4j) NEO=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), f"Python >= 3.11 required, found {sys.version}"'

echo "== 1/4 graph profile (77 diagnoses, 600 logic nodes, 11 operator kinds, Fig. 1 histogram)"
"$PY" scripts/graph_profile.py --graph "$GRAPH" --json "$OUT/graph_profile.json" | tail -n 2

echo "== 2/4 rule test: 788 positive / 298 refuted / 1530 near-miss cases against all 77 diagnoses, open + closed world"
"$PY" scripts/rule_test.py --graph "$GRAPH" --set "$GOLD" --json "$OUT/rule_test.json"

echo "== 3/4 Fig. 3 example case against 1.1 and 2.2"
"$PY" scripts/fig3_case.py --graph "$GRAPH" --case gold/fig3_case.json --codes 1.1,2.2 --json "$OUT/fig3_case.json" | head -n 2

if [ "$CYPHER" = 1 ]; then
  echo "== optional: regenerate the Neo4j Cypher from the bundle (1348 nodes / 1851 relationships)"
  "$PY" scripts/export_bundle_to_cypher.py --graph "$GRAPH" --vocab "${G}_vocab.json" --criteria "${G}_criteria.json" \
    --out "$OUT/neo4j_layered.cypher" --check --expected neo4j/expected_counts.json
  if cmp -s <(grep -v '^// Generator' "$OUT/neo4j_layered.cypher") <(grep -v '^// Generator' neo4j/neo4j_layered.cypher); then
    echo "   identical to neo4j/neo4j_layered.cypher (apart from the generator date line)"
  else
    echo "   DIFFERS from neo4j/neo4j_layered.cypher — inspect with diff"; exit 1
  fi
fi
if [ "$NEO" = 1 ]; then
  echo "== optional: Neo4j (docker compose)"
  (cd neo4j && ./load.sh)
fi

echo "== 4/4 check poster numbers"
"$PY" scripts/check_numbers.py --results "$OUT" --expected "$PKG/expected" --graph "$GRAPH" | tee "$OUT/check_numbers.md"
