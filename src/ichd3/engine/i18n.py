#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprachbündel (i18n) für Prompt-Prosa und sprachabhängige Entity-IDs
===================================================================

Zwei Bündel je Sprache:

1. ``Texts`` — die Prompt-Prosa (Section-Header, Ausgabe-Regel, Rückfragen-
   Templates, Kausal-Phrasen). **Deutsch (`_DE_TEXTS`) ist byte-identisch zu den
   bisherigen Literalen** in ``surface_forms.py``/``open_questions.py``/
   ``refine_pipeline.py`` — der byte-identity-Test schützt das.
2. ``Ids`` — sprachabhängige **Entity-IDs** (Ereignis-Träger, Raten-Symbole).
   Der deutsche und der englische ICHD-3-Graph haben getrennte Entity-ID-
   Namespaces (``attacke``/``kopfschmerz`` vs. ``attack``/``headache``); die
   Grounding-Brücke in ``patient_model`` braucht die Träger der jeweiligen Sprache.

Diagnose-Codes, Logic-IDs und Relations-Tokens sind in beiden Graphen identisch
und daher NICHT Teil dieser Bündel.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Texts:
    """Prompt-Prosa (sprachabhängig). Felder mit ``{...}`` sind ``str.format``-
    Templates; die Platzhalter werden von den Konsumenten gefüllt."""
    surface_header: str
    vocab_section_header: str
    reported_section_header: str
    output_rule: str
    oq_header: Tuple[str, str]
    rel_text: Dict[str, str]
    rel_fallback: str
    q_measure: str
    measure_fallback_name: str
    ground_disorder: str
    ground_hint: str
    ground_symptom: str
    ground_attribute: str
    neg_disorder: str
    neg_symptom: str
    neg_attribute: str
    ground_question: str
    q_causal: str
    causal_within: str
    temporal_anchor_fallback: str
    q_temporal_duration: str
    q_temporal_rel: str
    quantifier_scope_fallback: str
    q_quantifier: str
    q_reported: str
    q_period: str


@dataclass(frozen=True)
class Ids:
    """Sprachabhängige Entity-IDs (Graph-Namespace je Sprache)."""
    event_carriers: Tuple[str, ...]
    attack_count_carriers: Tuple[str, ...]
    rate_symbols: Dict[str, str]
    anchor_headache: str
    proven_attr: str


# ---------------------------------------------------------------------------
# Deutsch — byte-identisch zu den bisherigen Literalen (Korrektheitsmaß).
# ---------------------------------------------------------------------------
_DE_TEXTS = Texts(
    surface_header=(
        "=== SYNONYME/OBERFLÄCHENFORMEN (nur Mapping-Hilfe; gib AUSSCHLIESSLICH die id "
        "aus vocab_ref.json aus, NIE eine dieser Formen) ===\n"
        "Erkennst du im Text eine der folgenden Bezeichnungen/Abkürzungen (oft englisch), "
        "mappe sie auf die zugehörige id. Die Liste ist nicht erschöpfend.\n"
    ),
    vocab_section_header="\n\n=== VOKABULAR (vocab_ref.json, NUR diese ids) ===\n",
    reported_section_header="\n\n=== REPORTED-KATALOG (reported_catalog.json) ===\n",
    output_rule=("\n\n=== AUSGABE ===\nGib AUSSCHLIESSLICH EIN JSON-Objekt (Meta) "
                 "zurück — kein Markdown, kein Array, keine Vorrede. Jede evidence "
                 "ist wörtlicher Substring des Falltexts."),
    oq_header=("=== OFFENE RÜCKFRAGEN (beantworte NUR aus dem Falltext, mit "
               "wörtlichem Beleg; nichts erfinden) ===",
               "Diese Fakten würden eine naheliegende Diagnose bestätigen oder "
               "ausschließen. Liefere für belegbare Punkte den passenden "
               "Meta-Eintrag (findings/disorders/causal_relations/durations/"
               "counts/reported) mit evidence; nicht Belegbares weglassen."),
    rel_text={
        "developed_after": "begann in zeitlichem Zusammenhang mit dem Auftreten/Beginn von",
        "improves_with": "bessert sich mit Behandlung/Rückbildung von",
        "worsens_with": "verschlechtert sich parallel zu",
        "resolved_within": "bildete sich nach Behebung zurück von",
        "covaries_with": "schwankt parallel mit",
        "led_to_discovery_of": "führte zur Entdeckung von",
        "ipsilateral_to": "liegt auf derselben Seite wie",
        "contralateral_to": "liegt auf der gegenüberliegenden Seite von",
    },
    rel_fallback="steht in Relation '{rel}' zu",
    q_measure="Gibt es einen objektiven Messwert für '{nm}' (Vergleich {cmp} {val})?",
    measure_fallback_name="Messwert",
    ground_disorder="{neg}Hinweis auf {nm}{hint}",
    ground_hint=" (auch: {forms})",
    ground_symptom="{neg}{nm}",
    ground_attribute="{neg}{nm}",
    neg_disorder="KEIN ",
    neg_symptom="kein ",
    neg_attribute="NICHT ",
    ground_question="Belegt der Text: {body}?",
    q_causal="Belegt der Text, dass der Kopfschmerz {phrase} {dis}{within}?",
    causal_within=" innerhalb von {within} {unit}",
    temporal_anchor_fallback="die Attacke/der Kopfschmerz",
    q_temporal_duration="Wie lange dauert {anchor} (Dauer-/Zeitangabe im Text)?",
    q_temporal_rel="Belegt der Text die zeitliche Beziehung '{rel}' für {anchor}?",
    quantifier_scope_fallback="die Attacken",
    q_quantifier=("Wie häufig/wie viele {scope} nennt der Text "
                  "(Anzahl oder Rate je Zeitfenster)?"),
    q_reported="Belegt der Text das klinische Merkmal: '{nm}'?",
    q_period=("Verlauf episodisch oder chronisch? Aktive Periode/längste "
              "Remission (Dauerangaben)?"),
)

# ---------------------------------------------------------------------------
# Englisch — native Klinik-Prosa; Entity-Namen kommen aus dem englischen Graphen.
# ---------------------------------------------------------------------------
_EN_TEXTS = Texts(
    surface_header=(
        "=== SYNONYMS/SURFACE FORMS (mapping aid only; output ONLY the id from "
        "vocab_ref.json, NEVER one of these forms) ===\n"
        "If you recognise one of the following terms/abbreviations in the text, "
        "map it to the corresponding id. The list is not exhaustive.\n"
    ),
    vocab_section_header="\n\n=== VOCABULARY (vocab_ref.json, ONLY these ids) ===\n",
    reported_section_header="\n\n=== REPORTED CATALOG (reported_catalog.json) ===\n",
    output_rule=("\n\n=== OUTPUT ===\nReturn EXCLUSIVELY ONE JSON object (Meta) "
                 "— no Markdown, no array, no preamble. Every evidence is a "
                 "verbatim substring of the case text."),
    oq_header=("=== OPEN QUESTIONS (answer ONLY from the case text, with "
               "verbatim evidence; invent nothing) ===",
               "These facts would confirm or rule out a likely diagnosis. For "
               "provable points, provide the matching meta entry (findings/"
               "disorders/causal_relations/durations/counts/reported) with "
               "evidence; omit what cannot be supported."),
    rel_text={
        "developed_after": "began in temporal relation to the onset/start of",
        "improves_with": "improves with treatment/resolution of",
        "worsens_with": "worsens in parallel with",
        "resolved_within": "resolved after treatment of",
        "covaries_with": "covaries with",
        "led_to_discovery_of": "led to the discovery of",
        "ipsilateral_to": "is on the same side as",
        "contralateral_to": "is on the opposite side of",
    },
    rel_fallback="is in relation '{rel}' to",
    q_measure="Is there an objective measurement for '{nm}' (comparison {cmp} {val})?",
    measure_fallback_name="measurement",
    ground_disorder="{neg}evidence of {nm}{hint}",
    ground_hint=" (also: {forms})",
    ground_symptom="{neg}{nm}",
    ground_attribute="{neg}{nm}",
    neg_disorder="NO ",
    neg_symptom="no ",
    neg_attribute="NOT ",
    ground_question="Does the text document: {body}?",
    q_causal="Does the text document that the headache {phrase} {dis}{within}?",
    causal_within=" within {within} {unit}",
    temporal_anchor_fallback="the attack/the headache",
    q_temporal_duration="How long does {anchor} last (duration/time information in the text)?",
    q_temporal_rel="Does the text document the temporal relation '{rel}' for {anchor}?",
    quantifier_scope_fallback="the attacks",
    q_quantifier=("How often/how many {scope} does the text mention "
                  "(count or rate per time window)?"),
    q_reported="Does the text document the clinical feature: '{nm}'?",
    q_period=("Course episodic or chronic? Active period/longest remission "
              "(duration information)?"),
)

_DE_IDS = Ids(
    event_carriers=("attacke", "episode", "kopfschmerz"),
    attack_count_carriers=("attacke", "episode"),
    rate_symbols={"kopfschmerztag": "month"},
    anchor_headache="kopfschmerz",
    proven_attr="nachgewiesen",
)

_EN_IDS = Ids(
    # `headache_episode` ist Vokabular-id (Scope in 4.5.3) und wird von den Modellen
    # als Attackentraeger benutzt (27B: 76-79 % der Kopfschmerz-Dauern) — ohne Bruecke
    # verpufften diese Dauern/Zahlen (Report fehlende-fakten-ursachen.md).
    event_carriers=("attack", "episode", "headache", "headache_episode"),
    attack_count_carriers=("attack", "episode", "headache_episode"),
    rate_symbols={"headache_day": "month"},
    anchor_headache="headache",
    proven_attr="demonstrated_present",
)

_TEXTS = {"de": _DE_TEXTS, "en": _EN_TEXTS}
_IDS = {"de": _DE_IDS, "en": _EN_IDS}


def texts(lang: str = "de") -> Texts:
    """Prompt-Prosa-Bündel; unbekannte Sprache -> Deutsch (sicherer Default)."""
    return _TEXTS.get(lang, _DE_TEXTS)


def ids(lang: str = "de") -> Ids:
    """Entity-ID-Bündel; unbekannte Sprache -> Deutsch (sicherer Default)."""
    return _IDS.get(lang, _DE_IDS)
