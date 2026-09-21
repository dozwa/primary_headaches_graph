// Expected (see expected_counts.json): 1348 nodes, 1851 relationships.
MATCH (n) RETURN 'nodes' AS what, count(n) AS n;
MATCH ()-[r]->() RETURN 'relationships' AS what, count(r) AS n;
MATCH (n) UNWIND [l IN labels(n) WHERE l <> 'Ontology'] AS label RETURN label, count(*) AS n ORDER BY label;
MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS n ORDER BY relationship;
MATCH (:Logic)-[r:GROUNDS_ON {negated:true}]->() RETURN 'GROUNDS_ON negated' AS what, count(r) AS n;
