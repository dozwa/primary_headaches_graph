"""ICHD-3 Reasoning Pipeline — ein Paket, vier Schichten.

Subpackages:
  * engine     — Kern: dreiwertiger Reasoner, Patientenmodell, Vokabular,
                 Neo4j-Anbindung, SQLite-Storage, Extraktions-Runner
  * reasoners  — Varianten über der Engine: fuzzy, CF, IFS, specificity, reported
  * llm        — anbieter-agnostische LLM-Schnittstelle (anthropic/openai-kompatibel)
  * pipelines  — die vier End-zu-End-Architekturen (A–D) hinter einer Registry
  * runner     — config-getriebener Suite-Runner (YAML → Läufe → reasoner_results.db)

Dazu ``ichd3.cli`` (Entry-Point ``ichd3``) mit demo/vocab/extract/reason.
"""

__version__ = "0.1.0"
