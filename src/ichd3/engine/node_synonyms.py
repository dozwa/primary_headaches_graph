#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knoten-Synonym-Vokabular (Resolver-Schicht)
===========================================

Graph-INTERNE Subtyp->Eltern- bzw. Synonym-Mappings, die im Resolver
(``patient_model.meta_to_case``) angewandt werden — NICHT in den Graphen
geschrieben. Hintergrund: einige Extraktions-vocab-ids sind feiner/anders
benannt als die Disorder/Symptom-Entitäten der Graph-Kopplungs-/GROUND-Kriterien
(z.B. ``spontanes_liquorunterdruck`` vs. der Coupling-Ent ``liquorunterdruck``).
Ein korrekt extrahierter Subtyp entailt den gröberen Graph-Knoten -> er wird
zusätzlich unter dem Graph-Namen geführt, damit Kopplungen nicht "verpuffen".

Relationstypen:
  * ``subtype_of``  – gerichtet, NUR für status=present expandiert
                      (Subtyp->Eltern ist nur für Präsenz sound: aus dem Fehlen
                      des Subtyps folgt nicht das Fehlen des Elternknotens).
  * ``synonym_of``  – bidirektionale Äquivalenz, expandiert present UND absent.

Die Datei (``node_synonyms.json``) ist menschen-editierbar; fehlt/kaputt ->
``_FALLBACK`` (die historischen 4 Disorder-Aliasse), damit der Resolver bzw. die
Grounding-Gate (die ``meta_to_case`` als Dry-Run aufruft) nie crasht.
"""

from __future__ import annotations
import functools
import json
from pathlib import Path

from ichd3.engine.paths import repo_root
from typing import Dict, Iterator, List, Tuple, Union

# Default neben vocab_ref.json / reported_catalog.json / den Contracts.
DEFAULT_PATH = (repo_root()
                / "assets/datasets/reasoner_gold/selfextract_corpus/node_synonyms.json")

_KINDS = ("disorder", "symptom", "attribute")

# Sicheres Fallback == heutiges Verhalten (die 4 verifizierten Subtyp->Eltern-
# Disorder-Aliasse). Greift nur, wenn die Datei fehlt oder unlesbar ist.
_FALLBACK: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "disorder": {
        "spontanes_liquorunterdruck": [("liquorunterdruck", "subtype_of")],
        "virale_enzephalitis":        [("virale_meningitis_enzephalitis", "subtype_of")],
        "virale_meningitis":          [("virale_meningitis_enzephalitis", "subtype_of")],
        "multiple_sklerose":          [("ms_zentral", "subtype_of")],
    },
    "symptom": {},
    "attribute": {},
}


def load_synonyms(path: Union[str, Path] = DEFAULT_PATH
                  ) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """Lädt das Synonym-Vokabular als ``{kind: {from_id: [(to_id, relation), ...]}}``.

    Fehlende/kaputte Datei -> ``_FALLBACK`` (nie eine Exception nach außen, damit
    die Resolver-/Gate-Pfade robust bleiben).
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {k: dict(v) for k, v in _FALLBACK.items()}

    out: Dict[str, Dict[str, List[Tuple[str, str]]]] = {k: {} for k in _KINDS}
    for kind in _KINDS:
        for edge in raw.get(kind, []) or []:
            frm, to, rel = edge.get("from"), edge.get("to"), edge.get("relation")
            if not frm or not to or rel not in ("subtype_of", "synonym_of"):
                continue
            out[kind].setdefault(frm, []).append((to, rel))
    return out


@functools.lru_cache(maxsize=1)
def _cached_map() -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """Prozessweit einmal geladen. Tests umgehen den Cache via ``smap=``."""
    return load_synonyms()


def expand(node_id: str, kind: str, status: str,
           smap: Dict[str, Dict[str, List[Tuple[str, str]]]] | None = None
           ) -> Iterator[str]:
    """Liefert ``node_id`` und alle für ``status`` gültigen gemappten Ziele.

    ``synonym_of`` expandiert immer, ``subtype_of`` nur bei ``status == "present"``.
    Gibt stets zuerst die Original-id aus -> ohne Mapping == Identität.
    """
    m = smap if smap is not None else _cached_map()
    yield node_id
    for to_id, rel in m.get(kind, {}).get(node_id, ()):
        if rel == "synonym_of" or (rel == "subtype_of" and status == "present"):
            yield to_id


# --------------------------------------------------------------------------- #
# Kanonisierungs-Schicht: out-of-vocab Extraktions-IDs -> deutsche Graph-Ground-IDs
# --------------------------------------------------------------------------- #
# Eigene, von subtype_of/synonym_of GETRENNTE Ebene: jene mappen IN-vocab Entailment
# (Subtyp->Eltern, beide im kontrollierten Vokabular). Die Kanonisierung mappt dagegen
# OUT-of-vocab Schreibweisen — englische Ontologie-Namen (intracerebral_haemorrhage),
# Tippvarianten (subarachnoidblutung) — auf die deutsche Graph-ID. Sie ist eine reine
# UMBENENNUNG und wird im Resolver VOR ``expand`` angewandt; danach greift die Subtyp-
# Kette normal. Hintergrund: die Extraktion (deutsches Profil über engl. Falltexte)
# leakt teils englische Disorder-/Symptom-IDs, die das deutsche Graph-Grounding der
# Sekundär-Kausalkriterien nicht matchen -> Kopplung verpufft (μ=UNKNOWN).
ALIAS_PATH = (repo_root()
              / "assets/datasets/reasoner_gold/selfextract_corpus/entity_alias_de.json")
_ALIAS_KINDS = ("disorder", "symptom")


def load_aliases(path: Union[str, Path] = ALIAS_PATH) -> Dict[str, Dict[str, str]]:
    """Lädt die EN->DE-Alias-Map als ``{kind: {from_id: to_id}}``. Fehlende/kaputte
    Datei -> leere Maps (Kanonisierung == Identität, nie eine Exception nach außen)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {k: {} for k in _ALIAS_KINDS}
    return {k: {str(f): str(t) for f, t in (raw.get(k) or {}).items() if isinstance(t, str)}
            for k in _ALIAS_KINDS}


@functools.lru_cache(maxsize=1)
def _cached_alias() -> Dict[str, Dict[str, str]]:
    return load_aliases()


def canonicalize(node_id: str, kind: str,
                 amap: Dict[str, Dict[str, str]] | None = None) -> str:
    """Mappt eine out-of-vocab Extraktions-ID auf die deutsche Graph-Ground-ID
    (englische Onto-Namen, Tippvarianten). Ohne Eintrag == Identität (Passthrough)."""
    if not node_id:
        return node_id
    m = amap if amap is not None else _cached_alias()
    return m.get(kind, {}).get(node_id, node_id)
