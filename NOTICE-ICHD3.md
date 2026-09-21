# Notice on the ICHD-3 classification text

The International Classification of Headache Disorders, 3rd edition (ICHD-3; Headache
Classification Committee of the International Headache Society, *Cephalalgia* 2018;38(1):1–211,
doi:10.1177/0333102417738202) is copyright of the International Headache Society (IHS).

The IHS permits free reproduction of ICHD-3 "for scientific, educational or clinical uses by
institutions, societies or individuals"; **commercial** reproduction of any part requires the
Society's permission via Sage Publications Ltd (https://ichd-3.org/copyright/). This package is
distributed under exactly those terms — see `LICENSE-DATA` for the full statement and address.

Where ICHD-3 text appears in this package:

- `graph/ichd3_decomposition_v6_primary_criteria.json` — verbatim criterion texts, one entry per
  criterion / sub-criterion (523 entries for chapters 1–4); the `text_en` property in
  `neo4j/neo4j_layered.cypher` carries the same strings;
- `graph/ichd3_decomposition_v6_primary_vocab.json` — controlled vocabulary of grounding terms with
  short surface forms (identifiers and synonyms, not criterion prose);
- `graph/*_graph.json`, `gold/` — the formalization and the cases derived from the criteria.

Only `graph/*_graph.json` is needed to run the reasoner; it contains the logic structure and
grounding-term identifiers, not the criterion prose. The code of this package (`src/`, `scripts/`)
contains no ICHD-3 text and is licensed under MIT (`LICENSE-CODE`).
