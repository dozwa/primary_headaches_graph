#!/usr/bin/env bash
# Start Neo4j 5 via docker compose, load neo4j_layered.cypher, print label/relationship counts.
#   ./load.sh          load (idempotent: MERGE statements)
#   ./load.sh --down   stop and remove the container and its volume
set -euo pipefail
cd "$(dirname "$0")"
PASS="${NEO4J_PASSWORD:-password}"
if [ "${1:-}" = "--down" ]; then docker compose down -v; exit 0; fi
docker compose up -d
printf 'waiting for Neo4j'
for _ in $(seq 1 60); do
  if docker compose exec -T neo4j cypher-shell -u neo4j -p "$PASS" 'RETURN 1;' >/dev/null 2>&1; then echo ' — ready'; break; fi
  printf '.'; sleep 2
done
echo "loading neo4j_layered.cypher ($(grep -vc '^//' neo4j_layered.cypher) statements)"
docker compose exec -T neo4j cypher-shell -u neo4j -p "$PASS" --format plain < neo4j_layered.cypher >/dev/null
echo 'counts:'
docker compose exec -T neo4j cypher-shell -u neo4j -p "$PASS" --format plain < verify_counts.cypher
echo 'expected: see expected_counts.json (1348 nodes, 1851 relationships). Browser: http://localhost:7474'
