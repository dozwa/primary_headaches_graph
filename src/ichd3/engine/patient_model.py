#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Patientenmodell (LLM-Extraktionsschicht)
=============================================

Strukturierte Zwischenrepräsentation, die ein LLM aus einer klinischen Vignette
erzeugt und die der Reasoner (ichd3_reasoner.py) als `Case` konsumiert.

Kernidee — nur Belegbares ausgeben, sonst NICHTS:
    present  : Merkmal im Text explizit vorhanden      (mit wörtlichem Beleg)
    absent   : Merkmal im Text explizit verneint       (mit wörtlichem Beleg)
    (weggelassen) : Merkmal NICHT belegbar -> kein Eintrag (== UNKNOWN)

Das LLM gibt nur Befunde aus, die es am Text belegen kann. Es setzt NICHT alle
nicht gefundenen Merkmale auf "unknown" — Unbekanntheit entsteht durch das
Weglassen des Eintrags. Der Reasoner liest fehlende Fakten ohnehin als UNKNOWN
(Kleene-Logik). Würde Stillschweigen stattdessen als "absent" ausgegeben,
feuerten Negationskriterien (z.B. 1.2.1.2 "ohne Kopfschmerz") falsch — deshalb
ist "absent" nur bei ausdrücklicher Verneinung erlaubt, nie aus Stillschweigen.

Bausteine:
  * SCHEMA                  – JSON-Schema zur Validierung der LLM-Ausgabe
  * EXTRACTION_PROMPT       – System-/Aufgabenvertrag fürs LLM
  * VOCABULARY_EXPORT_CYPHER– erzeugt das erlaubte Vokabular AUS dem Graphen
  * meta_to_case()          – Adapter Meta-JSON -> Reasoner-Case
  * normalize_duration_nodes() – Einheiten-Normalisierung (h/min -> min)
"""

from __future__ import annotations
import json
import re
import sys
from typing import List

# Reasoner-Typen (Paket-intern)
from ichd3.engine.reasoner import Case, Reasoner, classify, selftest_graph, K, RuleGraph
# Resolver-Synonym-Schicht (Knoten-Vokabular; nicht im Graphen)
from ichd3.engine.node_synonyms import expand as _expand, canonicalize as _canon

# =============================================================================
# 1) JSON-SCHEMA  (Draft-2020-12-Stil; nutzbar für strukturierte LLM-Ausgaben)
# =============================================================================
# Nur belegbare Zustände. "unknown" gibt es bewusst NICHT: Unbekanntheit wird
# durch Weglassen des Eintrags ausgedrückt, nicht durch einen Status-Wert.
STATUS = {"enum": ["present", "absent"]}
# Vergleichsoperator einer Zahl: "unter 1 Tag/Monat" ist value 1 mit "<", nicht value 1.
# Ohne das Feld wurde die Schranke zum Punktwert und widerlegte 2.1/n3 (<1) — #31.
COMPARATOR = {"enum": ["<", "<=", "=", ">=", ">", None]}

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ICHD3MetaPatientModel",
    "type": "object",
    "required": ["vignette_id", "findings"],
    "additionalProperties": False,
    "properties": {
        "vignette_id": {"type": "string"},
        "language": {"enum": ["de", "en"]},
        # Symptome und (Symptom,Attribut)-Befunde, je dreiwertig + Beleg
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symptom", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "symptom": {"type": "string"},                 # Ontologie-id, z.B. 'aura'
                    "attribute": {"type": ["string", "null"]},     # Ontologie-id oder null
                    "status": STATUS,
                    "evidence": {"type": "string", "minLength": 1}, # wörtliches Zitat aus der Vignette
                    "attack_index": {"type": ["integer", "null"]}, # für spätere Pro-Attacke-Bindung
                },
            },
        },
        # Dauern – IMMER in Minuten normalisiert
        "durations": {
            "type": "array",
            "items": {
                "type": "object",
                # min_minutes ist PFLICHT (null erlaubt): ohne den Schluessel liess das
                # Modell die Zahl weg, obwohl sie im Belegsatz stand.
                "required": ["symptom", "min_minutes", "comparator", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "symptom": {"type": "string"},
                    "min_minutes": {"type": ["number", "null"]},
                    "max_minutes": {"type": ["number", "null"]},
                    "comparator": COMPARATOR,              # bezieht sich auf min_minutes
                    "status": STATUS,
                    "evidence": {"type": "string", "minLength": 1},
                },
            },
        },
        # Zählungen (z.B. Anzahl Attacken für den QUANTIFIER). "window" trennt die
        # kumulative Lebenszeit-Gesamtzahl ("lifetime", Default) von einer Rate
        # pro Zeitfenster ("per_period", mit "per": day/week/month) — eine
        # Tagesfrequenz ist KEINE Lebenszeit-Summe.
        "counts": {
            "type": "array",
            "items": {
                "type": "object",
                # value ist PFLICHT (null erlaubt). Gemessen (Gold 2, 3 472 Zaehlfakten):
                # mit optionalem value fehlte die Zahl in 70 % der Eintraege, obwohl sie
                # in 2 220 Faellen woertlich im zitierten Beleg stand.
                "required": ["symptom", "value", "comparator", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "symptom": {"type": "string"},
                    "value": {"type": ["integer", "null"]},
                    "comparator": COMPARATOR,
                    "window": {"enum": ["lifetime", "per_period"]},
                    "per": {"enum": ["day", "week", "month"]},
                    "status": STATUS,
                    "evidence": {"type": "string", "minLength": 1},
                },
            },
        },
        # Ursächliche Erkrankungen/Substanzen (Sekundärkopfschmerz, Disorder-Achse)
        "disorders": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["disorder", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "disorder": {"type": "string"},                 # Disorder-id, z.B. 'triptan'
                    "attribute": {"type": ["string", "null"]},      # z.B. 'uebergebrauch' / 'nachgewiesen'
                    "status": STATUS,
                    "evidence": {"type": "string", "minLength": 1},
                },
            },
        },
        # Verlaufsmuster des Kopfschmerzes (episodisch in Bouts vs. chronisch)
        "period": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"enum": ["episodic", "chronic"]},
                "active_period_days": {"type": ["number", "null"]},       # Bout-Länge (episodisch)
                "sustained_months": {"type": ["number", "null"]},         # Dauer des Musters
                "longest_remission_months": {"type": ["number", "null"]}, # längste Remission
                "evidence": {"type": "string", "minLength": 1},
            },
        },
        # Messwerte (comparison-GROUND, §3.2): numerischer Befund (z.B. Liquordruck)
        "measurements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["measurement", "value", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "measurement": {"type": "string"},          # Mess-Entität, z.B. 'liquordruck'
                    "value": {"type": ["number", "null"]},      # gemessener Zahlwert
                    "status": STATUS,
                    "evidence": {"type": "string", "minLength": 1},
                },
            },
        },
        # Kausale Kopplungen (Kausalitäts-Nachweis bei Sekundärkopfschmerz)
        "causal_relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["relation", "a", "b", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    # covaries_with stand im Graphen, fehlte aber hier — der Extraktor
                    # konnte es nie liefern. led_to_discovery_of / ipsilateral_to /
                    # contralateral_to kamen mit der Kontrakt-Erweiterung dazu
                    # (annotations_tool 0c4acb1); alle vier landen graph-seitig als
                    # CAUSAL_COUPLING und werden von case.causal generisch gematcht.
                    "relation": {"enum": ["developed_after", "resolved_within",
                                          "persists_beyond", "improves_with",
                                          "worsens_with", "provoked_by",
                                          "abolished_by", "covaries_with",
                                          "led_to_discovery_of",
                                          "ipsilateral_to", "contralateral_to"]},
                    "a": {"type": "string"},     # eine Entität (Symptom/Disorder), z.B. 'kopfschmerz'
                    "b": {"type": "string"},     # die andere, z.B. 'akute_rhinosinusitis'
                    "within": {"type": ["number", "null"]},  # Zeitschranke (für developed_after/resolved_within)
                    "unit": {"enum": ["minutes", "hours", "days"]},
                    "status": STATUS,
                    "evidence": {"type": "string", "minLength": 1},
                },
            },
        },
        # Temporale Relationen zwischen Symptomen (Felder in MINUTEN)
        "temporal_relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["rel", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    # Tokens AUS dem Graphen (t.rel, AP-G8 „Graph ist Quelle"):
                    # within/before/during sind die Katalog-Tokens der 10 fenster-
                    # tragenden Aura-/Timing-Knoten; die älteren Aliasse bleiben
                    # additiv, damit kein je gesehenes Token unrepräsentierbar wird.
                    "rel": {"enum": ["after", "overlaps", "concurrent",
                                     "succession", "gradual_spread",
                                     "accompanies_or_precedes_within",
                                     "within", "before", "during"]},
                    "from": {"type": ["string", "null"]},   # paarweise Relationen
                    "to": {"type": ["string", "null"]},
                    "sym": {"type": ["string", "null"]},    # sym-Relationen (succession/spread)
                    "within": {"type": ["number", "null"]}, # Minuten (after/accompanies)
                    "count": {"type": ["integer", "null"]}, # succession
                    "min": {"type": ["number", "null"]},    # gradual_spread (Minuten)
                    "status": STATUS,
                    "evidence": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

# =============================================================================
# 2) VOKABULAR-EXPORT  – die erlaubten ids kommen AUS dem Graphen, nicht erfunden
# =============================================================================
VOCABULARY_EXPORT_CYPHER = """
// Symptome (surface_en = Synonyme/Oberflächenformen, jetzt im Graphen gepflegt)
MATCH (s:Symptom)
  RETURN 'symptom' AS art, s.id AS id, s.name AS name, coalesce(s.surface_en, []) AS surface
UNION
// Attribute (surface_en = kurze Synonyme; text_en bleibt bewusst draußen — das ist
// Kriterien-Prosa, keine Oberflächenform)
MATCH (a:Attribute)
  RETURN 'attribute' AS art, a.id AS id, a.name AS name, coalesce(a.surface_en, []) AS surface
UNION
// ursächliche Erkrankungen/Substanzen (Sekundärkopfschmerz)
MATCH (x:Disorder)
  RETURN 'disorder' AS art, x.id AS id, x.name AS name, coalesce(x.surface_en, []) AS surface
UNION
// genutzte Temporal-Relationen (keine Oberflächenformen)
MATCH (t:Logic {kind:'TEMPORAL'})
  RETURN 'rel' AS art, t.rel AS id, t.rel AS name, [] AS surface
ORDER BY art, id;
""".strip()

# Migräne/Aura-Teilmenge (Stand der gesehenen Daten). MASSGEBLICH ist der
# Export oben — diese Liste dient als Startwert für den Prompt und muss bei
# Schema-Änderungen aus dem Graphen neu erzeugt werden.
KNOWN_VOCAB = {
    "symptoms": ["attacke", "aura", "schmerz"],
    "attributes": [
        "unilateral", "pulsierend", "intensitaet_moderat", "intensitaet_stark",
        "verstaerkung_durch_aktivitaet", "vermeidung_aktivitaet",
        "uebelkeit", "erbrechen", "photophobie", "phonophobie",
        "fully reversible", "visuell", "sensorisch", "motor",
        "hirnstamm", "retinal", "positiv",
    ],
    "rels": ["after", "overlaps", "concurrent", "succession",
             "gradual_spread", "accompanies_or_precedes_within",
             "within", "before", "during"],
    # Ursächliche Erkrankungen/Substanzen (kuratierte Teilmenge; maßgeblich ist
    # der Export aus dem Graphen, der ALLE Disorder-Knoten liefert).
    "disorders": ["akutmedikation", "triptan", "ergotamin", "opioid",
                  "acetylsalicylsaeure", "alkohol"],
}

# =============================================================================
# 3) LLM-EXTRAKTIONSVERTRAG
# =============================================================================
# Template mit Platzhaltern {symptoms}/{attributes}/{rels}. ichd3.engine.vocabulary.
# build_prompt(vocab) füllt es mit dem AUS DEM GRAPHEN exportierten Vokabular;
# EXTRACTION_PROMPT unten ist der Fallback mit der hartcodierten KNOWN_VOCAB.
EXTRACTION_PROMPT_TEMPLATE = """\
Du extrahierst aus einer klinischen Kopfschmerz-Vignette ein strukturiertes
Patientenmodell als JSON. Gib AUSSCHLIESSLICH gültiges JSON gemäß Schema aus,
ohne Vorrede, ohne Markdown.

NUR BELEGBARES AUSGEBEN (wichtigste Regel — nicht verletzen):
  - Gib NUR Befunde aus, die du direkt am Text der Vignette belegen kannst.
  - "present" nur, wenn der Text das Merkmal positiv beschreibt.
  - "absent"  nur, wenn der Text das Merkmal AUSDRÜCKLICH verneint
              (z.B. "keine Übelkeit", "kein begleitender Kopfschmerz").
  - Ist ein Merkmal NICHT erwähnt, LASS DEN EINTRAG WEG. Setze nicht erwähnte
    Merkmale NICHT auf irgendeinen Status und zähle nicht das Vokabular durch —
    fehlende Einträge gelten ohnehin als unbekannt.
  - Leite NIEMALS aus Stillschweigen "absent" ab. Im Zweifel: weglassen.
  Es gibt nur die zwei Status-Werte "present" und "absent".

VOKABULAR (nur diese ids verwenden; passt kein Merkmal, lass es weg — erfinde
keine ids):
  Symptome:   {symptoms}
  Attribute:  {attributes}
  Relationen: {rels}
  Erkrankungen/Substanzen (Disorder): {disorders}

URSÄCHLICHE ERKRANKUNG/SUBSTANZ (Sekundärkopfschmerz, Feld "disorders"):
  Beschreibt die Vignette einen Kopfschmerz, der auf eine andere Erkrankung oder
  eine Substanz zurückgeht (z.B. Medikamentenübergebrauch, Infektion, Gefäß-
  oder Stoffwechselstörung), trage die ursächliche Größe als "disorder"-Eintrag
  ein — mit dem passenden Attribut, das die Art des Belegs nennt (z.B.
  "uebergebrauch" bei Medikamentenübergebrauch, "nachgewiesen" bei gesicherter
  Diagnose). Beispiel: Triptan-Übergebrauch ->
  {{"disorder":"triptan","attribute":"uebergebrauch","status":"present","evidence":"…"}}.
  Nur present/absent mit Beleg; nichts Belegbares weglassen.

KAUSALITAETS-NACHWEIS (Sekundaerkopfschmerz, Feld "causal_relations"):
  Belegt die Vignette, dass der Kopfschmerz URSAECHLICH mit der Erkrankung/Substanz
  zusammenhaengt, trage die Kopplung ein (a/b sind die zwei beteiligten ids,
  Reihenfolge egal):
    - "developed_after" : KS entstand in zeitlichem Bezug zum Beginn der Stoerung
                          (optional within/unit, z.B. innerhalb 7 Tagen).
    - "resolved_within" : KS bildete sich nach Beseitigung/Abklingen zurueck
                          (within/unit angeben).
    - "persists_beyond" : KS besteht ueber die erwartete Dauer hinaus fort.
    - "worsens_with"/"improves_with" : KS verschlechtert/bessert sich parallel.
    - "provoked_by"/"abolished_by"   : KS wird ausgeloest/beseitigt durch X.
    - "covaries_with"   : KS-Staerke schwankt parallel mit X (beide Richtungen).
    - "led_to_discovery_of" : DER KOPFSCHMERZ fuehrte zur Entdeckung/Diagnose von X
                          ("die Abklaerung des Kopfschmerzes deckte X auf").
    - "ipsilateral_to"/"contralateral_to" : SEITENBEZUG -- der Schmerz liegt auf
                          derselben/der gegenueberliegenden Seite wie X
                          ("linksseitiger Kopfschmerz bei linksseitiger Dissektion").
                          Nur eintragen, wenn die Vignette die Seite BEIDER nennt
                          oder den Bezug ausdruecklich herstellt.
  Beispiel: {{"relation":"developed_after","a":"kopfschmerz","b":"akute_rhinosinusitis","within":7,"unit":"days","status":"present","evidence":"..."}}.
  Nur present/absent mit Beleg.

VERLAUFSMUSTER (Feld "period", v.a. trigemino-autonome Kopfschmerzen):
  Beschreibt die Vignette den zeitlichen Verlauf der Attacken-Perioden, trage
  ihn EINMAL ein:
    - mode "episodic": Attacken treten in Phasen/Bouts auf, getrennt durch
      Remissionen (>= 3 Monate). active_period_days = Länge einer aktiven Phase
      in Tagen; longest_remission_months = längste beschwerdefreie Zeit.
    - mode "chronic": Attacken ohne (oder mit nur kurzer, < 3 Monate) Remission
      über lange Zeit. sustained_months = bisherige Dauer; longest_remission_
      months = längste Remission (klein/0).
  Nur ausfüllen, was belegt ist; sonst Feld weglassen.

BELEGPFLICHT: Jeder Eintrag MUSS ein wörtliches Zitat aus der Vignette in
"evidence" tragen. Keine Paraphrase. Findest du kein Zitat, gib den Eintrag
nicht aus.

TRÄGERSYMBOL FÜR KOPFSCHMERZ-MERKMALE (entscheidend fürs Matching):
  Der Regelgraph modelliert den Kopfschmerz als ATTACKE und hängt deren
  Merkmale an das Symbol "{carrier}" (nicht an "{headache}"). Deshalb:
  - Merkmale einer einzelnen Kopfschmerzattacke — Lokalisation
    (unilateral/bilateral), Charakter (pulsierend/drückend/stechend),
    Intensität, Verstärkung/Vermeidung durch Aktivität, vegetative- und
    Begleitzeichen (Übelkeit, Erbrechen, Photophobie, Phonophobie, Lakrimation,
    nasale Kongestion, Ptosis, Miosis …) — hänge an "{carrier}".
  - Auch DAUER und ANZAHL der Attacken tragen das Symbol "{carrier}".
    Benutze GENAU diese id — keine Variante, keine Übersetzung, kein anderes
    Attacken-Wort aus dem Vokabular.
  - Aura-Merkmale (visuell, sensorisch, motor, hirnstamm, retinal,
    fully_reversible …) hänge an "aura"; die Dauer der Aura an "aura".
    Ist eine Aura beschrieben, gib ZUSÄTZLICH den bloßen Eintrag
    {{"symptom":"aura","status":"present"}} aus.
  - "{headache}" NUR für die bloße An-/Abwesenheit von Kopfschmerz
    selbst (z.B. "Aura ohne Kopfschmerz" -> {headache} absent), nicht als
    Träger der Attackenmerkmale.

ZAHLEN SIND PFLICHT (zweitwichtigste Regel):
  - Jede Zeitangabe im Text (Dauer einer Attacke/Aura, "seit 3 Monaten",
    "30 Minuten bis 7 Tage") gehört als ZAHL nach durations (min_minutes/
    max_minutes) — NIEMALS nur als Attribut-Phrase.
  - Jede Häufigkeit oder Anzahl ("fünf Attacken", "an 15 Tagen im Monat",
    "täglich") gehört als ZAHL nach counts (value). Ein counts-Eintrag ohne
    value ist wertlos: steht die Zahl im Text, trage sie ein; steht wirklich
    keine da, setze value auf null.
  - Zahlwörter zählen ("fünf" -> 5, "ein Dutzend" -> 12, "anderthalb Tage" -> 2160).
  - SCHRANKE (Feld "comparator", Pflicht): steht vor der Zahl eine Schranke, trage
    die Zahl SELBST ein und die Schranke als comparator — verschiebe nie die Zahl:
    "unter einem Tag im Monat" -> value 1, "<"; "höchstens 8 pro Tag" -> 8, "<=";
    "bis zu 8 pro Tag" ist der erreichte Spitzenwert -> 8, "=" (keine Schranke);
    "mehr als fünf Attacken" -> 5, ">"; "mindestens zehn" -> 10, ">=";
    "seit über 72 Stunden" -> min_minutes 4320, ">". Genaue oder ungefähre
    Angaben ("etwa 38 Stunden", "fünf Attacken") -> "=".
  - Steht KEINE Zahl im Text ("kurze Attacken", "häufig"), setze value bzw.
    min_minutes auf null — erfinde nie eine Zahl.

NORMALISIERUNG:
  - Alle Dauern und Zeitfenster in MINUTEN (z.B. "2 Tage" -> 2880).
  - Attribute IMMER an ihr Symptom binden (z.B. {{"symptom":"{carrier}",
    "attribute":"pulsierend"}}), nicht freistehend.
  - ZÄHLUNGEN (counts): unterscheide ZWEI Größen über das Feld "window" — sie
    sind NICHT austauschbar:
      A) "window":"lifetime" — KUMULATIVE Gesamtzahl der Ereignisse über die Zeit
         ("mindestens N Attacken gehabt"). Als GESICHERTE UNTERGRENZE, nie auf 1:
           * explizite Zahl ("mehr als fünf Attacken" -> 5, comparator ">").
           * aus Frequenz × Zeitspanne hochrechnen ("1-2x täglich über 6 Wochen"
             -> sicher >= 40; "2x/Woche seit 3 Monaten" -> ca. 24); Untergrenze
             ins value mit comparator ">=", die Rechnung ins evidence.
           * nur qualitativ ("seit Jahren rezidivierend"): konservative, aber
             realistische Untergrenze, die die Schilderung sicher hergibt, comparator ">=" — NICHT 1.
      B) "window":"per_period" mit "per": day/week/month — eine HÄUFIGKEIT/RATE
         pro Zeitfenster ("1-2 Attacken pro Tag", "an >=15 Tagen/Monat"). Trage
         den beobachteten Raten-Wert ein ("bis zu 8/Tag" = Spitzenwert -> 8, "="),
         eine echte Schranke als comparator, per = die Periode.
           * "{headache_day}" ist immer eine solche Rate (Tage/Monat: "täglich"
             -> 30, "an >15 Tagen/Monat" -> 15 mit ">").
      Eine geschilderte Tagesfrequenz IST eine Rate (B) — gib sie NICHT als
      lifetime-Zahl aus. Wo der Text BEIDES hergibt (Rate jetzt + viele über die
      Zeit), gib zwei Einträge: die Rate (B) UND die hochgerechnete lifetime-
      Untergrenze (A).
      * wirklich kein Anhaltspunkt: Eintrag WEGLASSEN (bleibt unbekannt), nie 1.

Antworte mit einem Objekt, das findings[], durations[], counts[],
temporal_relations[] und (falls zutreffend) disorders[], causal_relations[]
und period enthält.
"""

#: Version des Extraktions-Prompts — geht in den Extraktions-Cache-Fingerprint
#: (pipelines.context), damit eine Prompt-Aenderung nie gegen alte Extraktionen laeuft.
EXTRACTION_PROMPT_VERSION = 3

#: Englische Fassung fuer englische Graphen (Vokabular-Sprache = Prompt-Sprache).
#: Inhaltlich deckungsgleich mit EXTRACTION_PROMPT_TEMPLATE; Platzhalter identisch.
EXTRACTION_PROMPT_TEMPLATE_EN = """\
You extract a structured patient model as JSON from a clinical headache vignette.
Output EXCLUSIVELY valid JSON per the schema — no preamble, no Markdown.

ONLY WHAT IS SUPPORTED (most important rule — never violate it):
  - Emit only findings you can support directly in the vignette text.
  - "present" only if the text positively describes the feature.
  - "absent"  only if the text EXPLICITLY negates it
              (e.g. "no nausea", "no accompanying headache").
  - If a feature is NOT mentioned, OMIT the entry. Do not assign a status to
    unmentioned features and do not walk through the vocabulary — missing
    entries already count as unknown.
  - NEVER infer "absent" from silence. When in doubt: omit.
  Only the two status values "present" and "absent" exist.

VOCABULARY (use only these ids; if no feature fits, omit it — never invent ids):
  Symptoms:   {symptoms}
  Attributes: {attributes}
  Relations:  {rels}
  Disorders/substances: {disorders}

CAUSATIVE DISORDER/SUBSTANCE (secondary headache, field "disorders"):
  If the vignette describes a headache attributed to another disorder or a
  substance (e.g. medication overuse, infection, vascular or metabolic
  disorder), record the causative entity as a "disorder" entry — with the
  attribute naming the kind of evidence (e.g. "regularly_overused" for
  medication overuse, "{proven}" for an established diagnosis). Example:
  regular overuse of acute medication ->
  {{"disorder":"acute_medication","attribute":"regularly_overused","status":"present","evidence":"…"}}.
  Only present/absent with evidence; omit what cannot be supported.

CAUSAL EVIDENCE (secondary headache, field "causal_relations"):
  If the vignette shows that the headache is CAUSALLY linked to the
  disorder/substance, record the coupling (a/b are the two ids involved,
  order irrelevant):
    - "developed_after" : headache arose in temporal relation to the onset of
                          the disorder (optional within/unit, e.g. within 7 days).
    - "resolved_within" : headache resolved after removal/remission
                          (give within/unit).
    - "persists_beyond" : headache persists beyond the expected duration.
    - "worsens_with"/"improves_with" : headache worsens/improves in parallel.
    - "provoked_by"/"abolished_by"   : headache is triggered/abolished by X.
    - "covaries_with"   : headache intensity fluctuates with X (both directions).
    - "led_to_discovery_of" : THE HEADACHE led to the discovery/diagnosis of X
                          ("work-up of the headache revealed X").
    - "ipsilateral_to"/"contralateral_to" : SIDE RELATION — the pain is on the
                          same/opposite side as X ("left-sided headache with
                          left-sided dissection"). Only if the vignette names
                          BOTH sides or states the relation explicitly.
  Example: {{"relation":"developed_after","a":"{headache}","b":"acute_rhinosinusitis","within":7,"unit":"days","status":"present","evidence":"..."}}.
  Only present/absent with evidence.

COURSE PATTERN (field "period", mainly trigeminal autonomic cephalalgias):
  If the vignette describes the temporal course of attack periods, record it
  ONCE:
    - mode "episodic": attacks occur in phases/bouts separated by remissions
      (>= 3 months). active_period_days = length of an active phase in days;
      longest_remission_months = longest symptom-free period.
    - mode "chronic": attacks without (or with only short, < 3 months) remission
      over a long time. sustained_months = duration so far;
      longest_remission_months = longest remission (small/0).
  Fill in only what is supported; otherwise omit the field.

EVIDENCE MANDATORY: every entry MUST carry a verbatim quote from the vignette in
"evidence". No paraphrase. If you find no quote, do not emit the entry.

CARRIER SYMBOL FOR HEADACHE FEATURES (decisive for matching):
  The rule graph models the headache as an ATTACK and attaches its features to
  the symbol "{carrier}" (not to "{headache}"). Therefore:
  - Features of a single headache attack — location (unilateral/bilateral),
    quality (pulsating/pressing/stabbing), intensity, aggravation by or
    avoidance of activity, autonomic and accompanying signs (nausea, vomiting,
    photophobia, phonophobia, lacrimation, nasal congestion, ptosis, miosis …)
    — attach to "{carrier}".
  - DURATION and COUNT of attacks also carry the symbol "{carrier}". Use EXACTLY
    this id — no variant, no translation, no other attack word from the
    vocabulary.
  - Aura features (visual, sensory, motor, brainstem, retinal,
    fully_reversible …) attach to "aura"; the duration of the aura to "aura".
    If an aura is described, ALSO emit the bare entry
    {{"symptom":"aura","status":"present"}}.
  - "{headache}" ONLY for the mere presence/absence of headache itself (e.g.
    "aura without headache" -> {headache} absent), never as carrier of attack
    features.

NUMBERS ARE MANDATORY (second most important rule):
  - Every time span in the text (duration of an attack/aura, "for 3 months",
    "30 minutes to 7 days") goes as a NUMBER into durations (min_minutes/
    max_minutes) — NEVER only as an attribute phrase.
  - Every frequency or count ("five attacks", "on 15 days a month", "daily")
    goes as a NUMBER into counts (value). A counts entry without value is
    worthless: if the number is in the text, enter it; if there truly is none,
    set value to null.
  - Number words count ("five" -> 5, "a dozen" -> 12, "a day and a half" -> 2160).
  - BOUND (field "comparator", mandatory): if a bound precedes the number, enter
    the number ITSELF and the bound as comparator — never shift the number:
    "less than one day a month" -> value 1, "<"; "at most 8 per day" -> 8, "<=";
    "up to 8 per day" is the peak reached -> 8, "=" (not a bound);
    "more than five attacks" -> 5, ">"; "at least ten" -> 10, ">=";
    "for more than 72 hours" -> min_minutes 4320, ">". Exact or approximate
    figures ("about 38 hours", "five attacks") -> "=".
  - If the text gives NO number ("short attacks", "frequent"), set value or
    min_minutes to null — never invent a number.

NORMALISATION:
  - All durations and time windows in MINUTES (e.g. "2 days" -> 2880).
  - ALWAYS bind attributes to their symptom (e.g. {{"symptom":"{carrier}",
    "attribute":"pulsating_quality"}}), never free-standing.
  - COUNTS: distinguish TWO quantities via the field "window" — they are NOT
    interchangeable:
      A) "window":"lifetime" — CUMULATIVE total number of events over time
         ("has had at least N attacks"). As a SAFE LOWER BOUND, never 1:
           * explicit number ("more than five attacks" -> 5, comparator ">").
           * extrapolate from frequency × time span ("1-2x daily for 6 weeks"
             -> safely >= 40; "2x/week for 3 months" -> about 24); lower bound
             into value with comparator ">=", the calculation into evidence.
           * only qualitative ("recurring for years"): a conservative but
             realistic lower bound the description safely supports, comparator ">=" — NOT 1.
      B) "window":"per_period" with "per": day/week/month — a FREQUENCY/RATE per
         time window ("1-2 attacks per day", "on >=15 days/month"). Enter the
         observed rate value ("up to 8/day" = peak -> 8, "="), a real bound as
         comparator, per = the period.
           * "{headache_day}" is always such a rate (days/month: "daily" -> 30,
             "on >15 days/month" -> 15 with ">").
      A described daily frequency IS a rate (B) — do NOT emit it as a lifetime
      number. Where the text gives BOTH (rate now + many over time), emit two
      entries: the rate (B) AND the extrapolated lifetime lower bound (A).
      * truly no indication: OMIT the entry (stays unknown), never 1.

Answer with one object containing findings[], durations[], counts[],
temporal_relations[] and (if applicable) disorders[], causal_relations[]
and period.
"""

EXTRACTION_PROMPT = EXTRACTION_PROMPT_TEMPLATE.format(
    symptoms=", ".join(KNOWN_VOCAB["symptoms"]),
    attributes=", ".join(KNOWN_VOCAB["attributes"]),
    rels=", ".join(KNOWN_VOCAB["rels"]),
    disorders=", ".join(KNOWN_VOCAB["disorders"]),
    carrier="attacke", headache="kopfschmerz", headache_day="kopfschmerztag",
)

# =============================================================================
# 4) Einheiten-Normalisierung der Dauer-Knoten (h/min -> Minuten)
#    Einmal nach dem Laden des Graphen aufrufen, sonst scheitern Dauer-Tests
#    still (z.B. 1.1/B in Stunden vs. Fall-Dauer in Minuten).
# =============================================================================
UNIT_TO_MINUTES = {"sec": 1 / 60, "second": 1 / 60, "seconds": 1 / 60, "s": 1 / 60,
                   "min": 1, "minute": 1, "minutes": 1,
                   "h": 60, "hour": 60, "hours": 60,
                   "day": 1440, "days": 1440,
                   "week": 10080, "weeks": 10080,
                   "month": 43200, "months": 43200,
                   "year": 525600, "years": 525600}

def normalize_duration_nodes(g: RuleGraph) -> None:
    """Einheiten ALLER TEMPORAL-Knoten (min/max) nach Minuten skalieren — nicht nur
    'duration'. Auch relationale Schwellen ('after' within = max, 'gradual_spread'
    min) tragen eine `unit` (neo4j_layered emittiert sie); ohne Normalisierung würde
    z.B. eine Stunden-Frist roh gegen die Minuten-Falldaten geprüft. Zähl-Schwellen
    ohne Zeiteinheit (succession.count) haben unit=None -> Faktor 1 -> unverändert."""
    for nid, kind in g.kind.items():
        if kind != "TEMPORAL":
            continue
        p = g.props.get(nid, {})
        f = UNIT_TO_MINUTES.get((p.get("unit") or "minute").lower(), 1)
        if f != 1:
            if p.get("min") is not None:
                p["min"] = p["min"] * f
            if p.get("max") is not None:
                p["max"] = p["max"] * f
            p["unit"] = "minute"

# =============================================================================
# 5) ADAPTER  Meta-JSON -> Reasoner-Case
# =============================================================================
# Grounding-Brücke: Der Graph benennt DAS Kopfschmerz-Ereignis je nach Störung
# verschieden — "attacke" (Migräne, TAC), "episode" (Spannungskopfschmerz),
# "kopfschmerz" (mehrere Kapitel-4-Diagnosen). Eine diagnose-agnostische
# Extraktion kann nicht wissen, welcher Träger gefragt ist. Deshalb spiegeln wir
# Ereignis-Präsenz, -Attribute und -Dauer über diese Synonyme, damit Kriterien
# greifen, egal welchen Träger der Graph nutzt. (Das LLM lernt im Prompt, primär
# "attacke" zu verwenden; diese Spiegelung ist das deterministische Sicherheitsnetz.)
EVENT_CARRIERS = ("attacke", "episode", "kopfschmerz")
# Attacken-/Episodenanzahl sind austauschbar; "kopfschmerztag" (Tage/Monat) ist
# eine ANDERE Größe und wird bewusst nicht mitgespiegelt.
_ATTACK_COUNT_CARRIERS = ("attacke", "episode")
# Symbole, die per Konvention bereits eine RATE sind (Wert = Häufigkeit pro
# Periode), nicht eine kumulative Lebenszeit-Zahl. Symbol -> Standardperiode.
_RATE_SYMBOLS = {"kopfschmerztag": "month"}

#: Offene Grenze eines Vergleichsoperators ("unter 1" = [0, 1-EPS]). Die Graph-Schwellen
#: sind ganze Zahlen bzw. Minuten; EPS verschiebt nur den Randwert.
_OPEN_EPS = 1e-6

#: Wie eine Ereignisdauer (attack/episode/...) auf den KOPFSCHMERZ-Traeger kommt, wenn der
#: keine eigene Dauer hat. "copy": Punktwert kopiert (bis #31 — kippte 3.4 ">3 Monate" und
#: 1.4.1 ">72 h" auf FALSE, 83 von 217 falschen Widerlegungen beim 27B); "lower_bound": der
#: Kopfschmerz dauert mindestens so lange wie die Attacke, [lo, offen]; "none": gar nicht.
HEADACHE_DURATION_FROM_EVENT = "lower_bound"


#: Minuten-Angaben, die das Modell als Sekunden eintraegt ("gut sechseinhalb Minuten" -> 390 statt
#: 6,5). Lauf #32: 27B 7,5 %, 9B 11 % der Minuten-Dauern; kippte die SUNCT-Codes (3.3/n12 <= 10 min).
#: Korrigiert wird NUR, wenn der Beleg eine Minuten-Zahl n nennt, der Wert genau 60*n ist und keine
#: Zahl+Einheit des Belegs den Wert korrekt erklaert ("2 Minuten ... 2 Stunden" bleibt 120).
FIX_MINUTES_AS_SECONDS = True

_NUM_WORDS = {
    "ein": 1, "eins": 1, "eine": 1, "einer": 1, "einem": 1, "einen": 1, "a": 1, "an": 1, "one": 1,
    "zwei": 2, "two": 2, "drei": 3, "three": 3, "vier": 4, "four": 4, "fünf": 5, "fuenf": 5,
    "five": 5, "sechs": 6, "six": 6, "sieben": 7, "seven": 7, "acht": 8, "eight": 8, "neun": 9,
    "nine": 9, "zehn": 10, "ten": 10, "elf": 11, "eleven": 11, "zwölf": 12, "twelve": 12,
    "fünfzehn": 15, "fifteen": 15, "zwanzig": 20, "twenty": 20, "dreißig": 30, "thirty": 30,
    "vierzig": 40, "forty": 40, "fünfzig": 50, "fifty": 50, "sechzig": 60, "sixty": 60,
    "neunzig": 90, "ninety": 90, "halbe": 0.5, "halben": 0.5, "half": 0.5, "anderthalb": 1.5,
    "dreiviertel": 0.75,
}
_UNIT_MINUTES = (
    (r"sekunden?|seconds?|sek\.?|secs?", 1 / 60), (r"minuten?|minutes?|min\.?|mins?", 1),
    (r"stunden?|std\.?|hours?|hrs?", 60), (r"tagen?|tage|days?", 1440),
    (r"wochen?|weeks?", 10080), (r"monaten?|monate|months?", 43200),
    (r"jahren?|jahre|years?", 525600),
)
_NUM_TOKEN = r"(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)"


def _word_number(tok: str):
    t = tok.lower()
    if re.fullmatch(r"\d+(?:[.,]\d+)?", t):
        return float(t.replace(",", "."))
    if t in _NUM_WORDS:
        return float(_NUM_WORDS[t])
    if t.endswith("einhalb") and t[:-7] in _NUM_WORDS:          # "sechseinhalb" -> 6,5
        return _NUM_WORDS[t[:-7]] + 0.5
    m = re.fullmatch(r"(\w+?)und(\w+)", t)                      # "fünfundvierzig" -> 45
    if m and m.group(1) in _NUM_WORDS and m.group(2) in _NUM_WORDS and _NUM_WORDS[m.group(2)] >= 20:
        return float(_NUM_WORDS[m.group(1)] + _NUM_WORDS[m.group(2)])
    return None


def _unit_numbers(evidence: str) -> list:
    """[(Minutenfaktor, Zahl)] fuer jede Zahl (oder Spanne) direkt vor einem Einheitswort."""
    out = []
    for pat, factor in _UNIT_MINUTES:
        rx = rf"(?<!\w)({_NUM_TOKEN}(?:\s*(?:-|–|bis|to|oder|or)\s*{_NUM_TOKEN})?)\s*(?:{pat})(?!\w)"
        for m in re.finditer(rx, evidence, re.I):
            for tok in re.findall(_NUM_TOKEN, m.group(1), re.I):
                n = _word_number(tok)
                if n is not None:
                    out.append((factor, n))
    return out


def _minutes_as_seconds(value, evidence) -> bool:
    """True, wenn `value` eine als Sekunden eingetragene Minuten-Angabe des Belegs ist."""
    if value is None or not evidence:
        return False
    v = float(value)
    cands = _unit_numbers(evidence)
    close = lambda a, b: abs(a - b) <= 0.02 * max(1.0, abs(b))
    if any(close(v, n * f) for f, n in cands):
        return False                                   # Wert ist korrekt umgerechnet
    return any(f == 1 and n > 0 and close(v, 60 * n) for f, n in cands)


def _bound(v, comp):
    """(lo, hi) einer Zahl mit Vergleichsoperator; None fuer "="/fehlend (Punktwert)."""
    if comp == ">":
        return (v + _OPEN_EPS, None)
    if comp == ">=":
        return (v, None)
    if comp == "<":
        return (0, v - _OPEN_EPS)
    if comp == "<=":
        return (0, v)
    return None


#: Merkmale, die in ICHD-3 Kapitel 1.2 **Aura**-Merkmale sind: der Graph verankert
#: sie an ``aura`` (alle 1.2.x-GROUNDS_ON tragen ``aura`` als Traeger), die
#: Extraktion liefert sie jedoch haeufig an ``headache``/``attack`` (gemessen:
#: vertigo {headache 9, attack 3}, diplopia {headache 21, attack 1}). Die Axiome
#: der Aura-Familie suchen dann am falschen Anker und bleiben wirkungslos.
_AURA_FEATURES = frozenset({
    "visual", "sensory", "speech_language_disturbance", "motor", "retinal",
    "brainstem_symptom", "dysarthria", "vertigo", "tinnitus", "hypacusis",
    "diplopia", "ataxia", "decreased_consciousness", "motor_weakness",
    "monocular_visual_field_defect",
})

#: Traeger, von denen ein Aura-Merkmal auf ``aura`` umgeleitet wird — je Sprache.
_AURA_SOURCE_CARRIERS = {"headache", "attack", "kopfschmerz", "attacke"}


def _route_aura_features(meta: dict, _ids) -> dict:
    """Aura-Merkmale auf den Traeger ``aura`` umleiten — **gerichtet**, nur findings.

    Warum nicht ueber ``EVENT_CARRIERS``: diese Menge ist eine *Aequivalenzklasse*
    ("dasselbe Ereignis, andere Bezeichnung") und ``_bridge_carriers`` spiegelt sie
    **bidirektional**, inklusive der absent-Zweige. ``aura`` dort aufzunehmen macht
    aus einem extrahierten "keine Aura" ein "kein Kopfschmerz" und kostete in der
    Messung 115 Verdikte (18 Codes, 7 Vignetten fielen von FALSE auf UNKNOWN
    zurueck; netto -26 ohne die Axiome). Aura ist ein *Teil* der Attacke, kein
    Synonym fuer sie — die Umleitung muss gerichtet und auf findings beschraenkt
    bleiben.

    Gemessene Wirkung (500 reale Vignetten x 276 Diagnosen, Live-Graph
    ``bolt://localhost:7691``): allein 0 Verdikt-Aenderungen, gemeinsam mit den
    9 ``subtype_of``-Axiomen der Aura-Familie **+179** neu ausgeschlossene
    Diagnosen (1.2.1/1.2.1.1/1.2.1.2 je 58, 1.2.2 5) und **0** Rueckschritte.
    Beide Korrekturen sind einzeln wirkungslos und nur gemeinsam wirksam.
    """
    fs = meta.get("findings")
    if not fs:
        return meta
    hits = [f for f in fs
            if f.get("attribute") in _AURA_FEATURES
            and f.get("symptom") in _AURA_SOURCE_CARRIERS]
    if not hits:
        return meta                      # haeufiger Fall: nichts kopieren
    meta = dict(meta)
    meta["findings"] = [
        (dict(f, symptom="aura") if f in hits else f) for f in fs
    ]
    return meta


def _bridge_carriers(c: Case, EVENT_CARRIERS=EVENT_CARRIERS,
                     _ATTACK_COUNT_CARRIERS=_ATTACK_COUNT_CARRIERS,
                     anchor_headache: str = "kopfschmerz") -> None:
    cs = set(EVENT_CARRIERS)
    pres_attrs = {a for (s, a) in c.attrs_present if s in cs}
    # Das Ereignis ist präsent, wenn ein Träger present ist ODER Merkmale/Dauer/
    # Anzahl an einem Träger hängen — eine Attacke MIT Merkmalen existiert. Der
    # GROUND-Anker (_eval_ground) prüft die Symptom-Präsenz, nicht nur das
    # Attribut; ohne diese Ableitung bliebe jeder Attribut-GROUND UNKNOWN.
    event_present = (
        bool(c.present & cs) or bool(pres_attrs)
        or any(s in c.durations for s in EVENT_CARRIERS)
        or any(s in c.counts or s in c.count_bounds for s in _ATTACK_COUNT_CARRIERS)
    )
    if event_present:
        c.present |= cs
    elif c.absent & cs:
        c.absent |= cs
    # Attribute des Ereignisses (present-Spiegelung vor absent)
    for a in pres_attrs:
        for s in EVENT_CARRIERS:
            c.attrs_present.add((s, a))
    for a in {a for (s, a) in c.attrs_absent if s in cs} - pres_attrs:
        for s in EVENT_CARRIERS:
            c.attrs_absent.add((s, a))
    # Attackendauer
    src = next((s for s in EVENT_CARRIERS if s in c.durations), None)
    if src is not None:
        d0 = c.durations[src]
        for s in EVENT_CARRIERS:
            if s in c.durations:
                continue
            if s == anchor_headache and src != anchor_headache:
                if HEADACHE_DURATION_FROM_EVENT == "none":
                    continue
                if HEADACHE_DURATION_FROM_EVENT == "lower_bound":
                    c.durations[s] = (d0[0], None)
                    continue
            c.durations[s] = d0
    # Attacken-/Episodenanzahl (ohne kopfschmerztag)
    cnts = [c.counts[s] for s in _ATTACK_COUNT_CARRIERS if s in c.counts]
    if cnts:
        for s in _ATTACK_COUNT_CARRIERS:
            c.counts.setdefault(s, max(cnts))
    cb = [c.count_bounds[s] for s in _ATTACK_COUNT_CARRIERS if s in c.count_bounds]
    if cb:
        for s in _ATTACK_COUNT_CARRIERS:
            if s not in c.counts:
                c.count_bounds.setdefault(s, cb[0])
    # Attacken-/Episodenraten je Periode ebenso spiegeln (attacke<->episode)
    rate_pers = {p for (s, p) in c.rates if s in _ATTACK_COUNT_CARRIERS}
    for per in rate_pers:
        vals = [c.rates[(s, per)] for s in _ATTACK_COUNT_CARRIERS if (s, per) in c.rates]
        for s in _ATTACK_COUNT_CARRIERS:
            c.rates.setdefault((s, per), max(vals))
    for per in {p for (s, p) in c.rate_bounds if s in _ATTACK_COUNT_CARRIERS}:
        bs = [c.rate_bounds[(s, per)] for s in _ATTACK_COUNT_CARRIERS if (s, per) in c.rate_bounds]
        for s in _ATTACK_COUNT_CARRIERS:
            if (s, per) not in c.rates:
                c.rate_bounds.setdefault((s, per), bs[0])
    # Sustained-Hochrechnung: ein über sustained_months gehaltenes monatliches Frequenz-
    # muster impliziert eine kumulative Lifetime-UNTERGRENZE (Rate × Monate). Erfüllt
    # "mindestens N Attacken/Episoden" (z.B. 1.3/B ≥5) bei chronischen Mustern, die der
    # Text nur als Frequenz + Dauer hergibt (genau die im Kontrakt geforderte Hochrechnung,
    # hier deterministisch). Lower-Bound -> nur erhöhen, nie eine größere explizite Zahl
    # überschreiben. Basis: Kopfschmerztage/Monat (jeder KS-Tag >=1 Ereignis) oder Attacken-
    # rate/Monat.
    # Praesenz aus Belegen fuer ALLE Symptome: ein Symptom mit present-Attribut,
    # Dauer oder Zahl existiert. Bisher galt das nur fuer die Ereignistraeger —
    # `aura` mit visual:present und 32 min Dauer blieb "unbekannt" (Gold 2: in
    # 1 228 Aura-Faellen kein einziger blosser aura:present-Eintrag, 91 % nur
    # Attribute/Dauer). Ein explizites absent bleibt absent.
    implied = ({s for (s, _a) in c.attrs_present} | set(c.durations) | set(c.counts)
               | {s for (s, _p) in c.rates} | set(c.count_bounds)
               | {s for (s, _p) in c.rate_bounds})
    c.present |= {s for s in implied if s and s not in c.absent}
    # Aura-Dauer: der Graph fragt sie an `aura` (1.2/n13) UND `aura_symptom`
    # (1.2/n15), Gold und Extraktion liefern nur eine der beiden Formen.
    if "aura" in c.durations and "aura_symptom" not in c.durations:
        c.durations["aura_symptom"] = c.durations["aura"]
    elif "aura_symptom" in c.durations and "aura" not in c.durations:
        c.durations["aura"] = c.durations["aura_symptom"]
    sm = (c.period or {}).get("sustained_months")
    if sm:
        monthly = max([c.rates.get((s, "month"), 0)
                       for s in (("kopfschmerztag",) + tuple(_ATTACK_COUNT_CARRIERS))] + [0])
        implied = int(monthly * sm)
        if implied >= 1:
            for s in _ATTACK_COUNT_CARRIERS:
                if implied > c.counts.get(s, 0):
                    c.counts[s] = implied


# Hart indizierte Pflicht-Schlüssel je Listenfeld (die Keys, auf die
# ``meta_to_case`` mit ``x["..."]`` zugreift). Fehlt einer, ist der Eintrag
# unbrauchbar und würde einen KeyError auslösen → verwerfen.
_HARD_KEYS = {
    "findings": ("symptom", "status"),
    "durations": ("symptom", "status"),
    "counts": ("symptom", "status"),
    "temporal_relations": ("status",),
    "measurements": ("measurement",),
    "disorders": ("disorder", "status"),
    "causal_relations": ("status", "relation"),
}


def _prune_meta_lists(meta: dict) -> dict:
    """Kopie des Meta, in der Listeneinträge ohne ihre hart indizierten
    Pflicht-Schlüssel entfernt sind. No-op auf schema-konformer Ausgabe."""
    if not isinstance(meta, dict):
        return {}
    out = dict(meta)
    for field, keys in _HARD_KEYS.items():
        vals = meta.get(field)
        if not isinstance(vals, list):
            continue
        out[field] = [x for x in vals if isinstance(x, dict)
                      and all(x.get(k) not in (None, "") for k in keys)]
    return out


#: Symbol-ids, die die Modelle auf dem ENGLISCHEN Graphen erfinden (aus
#: "Kopfschmerzattacke"; 9B: ~80 % aller Zaehleintraege, auch auf englischem Text) und
#: die in keinem v6-Vokabular stehen. Reine Umbenennung auf die Graph-id — Kontroll-
#: rechnung: 300 Gold-Skelette mit attack->attacke fielen von 300 auf 4 TRUE.
_EN_SYMBOL_ALIASES = {"attacke": "attack", "kopfschmerzattacke": "attack",
                      "kopfschmerz": "headache", "kopfschmerztag": "headache_day"}


def _alias_symbols(meta: dict, aliases: dict) -> dict:
    """Symbol-ids in findings/durations/counts/temporal_relations umbenennen (Kopie)."""
    def fix(x: dict, keys) -> dict:
        if any(x.get(k) in aliases for k in keys):
            x = dict(x)
            for k in keys:
                if x.get(k) in aliases:
                    x[k] = aliases[x[k]]
        return x
    out = dict(meta)
    for ax in ("findings", "durations", "counts"):
        if meta.get(ax):
            out[ax] = [fix(x, ("symptom",)) for x in meta[ax]]
    if meta.get("temporal_relations"):
        out["temporal_relations"] = [fix(x, ("from", "to", "sym")) for x in meta["temporal_relations"]]
    return out


def meta_to_case(meta: dict, lang: str = "de") -> Case:
    from ichd3.engine.i18n import ids
    _ids = ids(lang)
    if lang == "en":
        meta = _alias_symbols(meta, _EN_SYMBOL_ALIASES)
    rate_symbols = _ids.rate_symbols
    # Englischer Graph: die Gold-/Extraktions-IDs SIND die Graph-IDs. Alias-
    # (entity_alias_de.json, EN->DE-Umbenennung) und Synonym-Schicht gehören zum
    # deutschen Graphen; angewandt würden sie englische IDs (headache,
    # bacterial_meningitis, ...) auf nicht existente DE-Knoten umschreiben.
    if lang == "en":
        canon = lambda ident, kind: ident
        expand = lambda ident, kind, status: (ident,)
    else:
        canon, expand = _canon, _expand
    c = Case(name=meta.get("vignette_id", "vignette"))
    # Defensiv: Listeneinträge verwerfen, denen ein hart indizierter Pflicht-
    # Schlüssel fehlt. Schema-konforme (guided-decoding) Ausgabe hat sie immer →
    # unverändert; nur freie json_object-Ausgabe (z. B. DeepSeek-Fallback) kann sie
    # weglassen und würde sonst hier mit KeyError abstürzen.
    meta = _prune_meta_lists(meta)
    meta = _route_aura_features(meta, _ids)
    for f in meta.get("findings", []):
        sym, attr, st = canon(f["symptom"], "symptom"), f.get("attribute"), f["status"]
        if attr is None:
            if st == "present":
                for s in expand(sym, "symptom", "present"):
                    c.present.add(s)
            elif st == "absent":
                for s in expand(sym, "symptom", "absent"):
                    c.absent.add(s)
        else:
            if st == "present":
                for s in expand(sym, "symptom", "present"):
                    c.attrs_present.add((s, attr))
            elif st == "absent":
                for s in expand(sym, "symptom", "absent"):
                    c.attrs_absent.add((s, attr))
        # nichts/weggelassen -> kein Eintrag (bleibt UNKNOWN im Reasoner)
    for d in meta.get("durations", []):
        if d["status"] == "present" and d.get("min_minutes") is not None:
            # max kann explizit null sein (offenes Ende/Punktwert) -> auf min
            # zurückfallen; .get(key, default) greift bei explizitem null nicht.
            mn, mx = d["min_minutes"], d.get("max_minutes")
            if FIX_MINUTES_AS_SECONDS and _minutes_as_seconds(mn, d.get("evidence")):
                mn = mn / 60
                mx = mx / 60 if mx is not None else None
            b = _bound(mn, d.get("comparator"))
            if b is not None:
                c.durations[d["symptom"]] = b
                continue
            if mx is None:
                mx = mn
            c.durations[d["symptom"]] = (mn, mx)
    for n in meta.get("counts", []):
        if n["status"] != "present" or n.get("value") is None:
            continue
        sym, val = n["symptom"], n["value"]
        per = n.get("per")
        b = _bound(val, n.get("comparator"))
        # Rate (per_period) vs. kumulative Lebenszeit-Zahl trennen. Ein Symbol,
        # das per Konvention bereits eine Rate IST (kopfschmerztag = Tage/Monat),
        # landet auch ohne explizites Fenster im Raten-Eimer (Rückwärtskompat.).
        if n.get("window") == "per_period" or sym in rate_symbols:
            key = (sym, per or rate_symbols.get(sym))
            if b is None:
                c.rates[key] = val
            else:
                c.rate_bounds[key] = b
        else:
            if b is None:
                c.counts[sym] = val
            else:
                c.count_bounds[sym] = b
    for r in meta.get("temporal_relations", []):
        fact = {k: r[k] for k in ("rel", "from", "to", "sym", "within", "count", "min")
                if r.get(k) is not None}
        if r["status"] == "present":
            c.rels.append(fact)
        elif r["status"] == "absent":
            c.rels_absent.append(fact)
        # weggelassene Relation -> weder rels noch rels_absent (UNKNOWN im Reasoner)
    per = meta.get("period")
    # Auch ohne 'mode' laden: eine Erhaltungsdauer (sustained_months) ohne
    # episodisch/chronisch-Klassifikation speist den fenster-bewussten QUANTIFIER
    # ('>=X Tage/Monat über >=N Monate'); PERIOD_PATTERN-Knoten bleiben mode-bedingt.
    if isinstance(per, dict) and any(per.get(k) is not None for k in (
            "mode", "active_period_days", "sustained_months", "longest_remission_months")):
        c.period = {
            "mode": per.get("mode"),
            "active_period_days": per.get("active_period_days"),
            "sustained_months": per.get("sustained_months"),
            "longest_remission_months": per.get("longest_remission_months"),
        }
    # Disorder/Symptom-Vokabular-Alignment: einige Extraktions-vocab-ids sind feiner/
    # anders benannt als die Entitäten der Graph-Kopplungs-/GROUND-Kriterien.
    #  (1) Kanonisierung (_canon): out-of-vocab Schreibweisen (englische Onto-Namen,
    #      Tippvarianten) -> deutsche Graph-ID, als reine Umbenennung VOR _expand.
    #  (2) Synonym-Schicht (_expand): führt einen korrekt extrahierten Subtyp/Synonym
    #      zusätzlich unter dem Graph-Namen, damit Kopplungen nicht "verpuffen".
    #      subtype_of expandiert NUR bei present (Subtyp->Eltern nur für Präsenz sound);
    #      synonym_of bidirektional.
    # BEIDE Kausal-Entitäten werden normalisiert UND expandiert: die a-Seite (Kopf-
    # schmerz) wurde zuvor NICHT durch _expand geführt -> headache verpuffte trotz Alias.
    for x in meta.get("causal_relations", []):
        a_ent = canon(x.get("a"), "symptom")
        b_ent = canon(x.get("b"), "disorder")
        wmin = None
        if x.get("within") is not None:
            wmin = x["within"] * UNIT_TO_MINUTES.get((x.get("unit") or "minute").lower(), 1)
        st_c = x["status"]
        if st_c in ("present", "absent"):
            a_opts = list(expand(a_ent, "symptom", st_c)) if a_ent else [None]
            b_opts = list(expand(b_ent, "disorder", st_c)) if b_ent else [None]
            sink = c.causals if st_c == "present" else c.causals_absent
            for aa in a_opts:
                for bb in b_opts:
                    ents = frozenset(e for e in (aa, bb) if e)
                    sink.append({"relation": x["relation"], "ents": ents, "within_minutes": wmin})
    for x in meta.get("disorders", []):
        dis, attr, st = canon(x["disorder"], "disorder"), x.get("attribute"), x["status"]
        if st == "present":
            for dd in expand(dis, "disorder", "present"):  # ein belegtes Attribut impliziert die Substanz
                c.disorders_present.add(dd)
                if attr is not None:
                    c.attrs_present.add((dd, attr))
        elif st == "absent":
            for dd in expand(dis, "disorder", "absent"):
                if attr is None:
                    c.disorders_absent.add(dd)
                else:
                    c.attrs_absent.add((dd, attr))
    # Messwerte (comparison-GROUND, §3.2): ent -> Zahlwert; der Reasoner vergleicht
    # ihn per measure() gegen die Knoten-Schwelle. Ohne diesen Loader bliebe jeder
    # numerische Vergleich UNKNOWN (Messwert nie eingelesen).
    for m in meta.get("measurements", []):
        if m.get("status", "present") == "present" and m.get("value") is not None:
            c.measurements[m["measurement"]] = m["value"]
    _bridge_carriers(c, _ids.event_carriers, _ids.attack_count_carriers, _ids.anchor_headache)
    return c


# Listenfelder des Meta + ihr Dedup-Schlüssel (ohne 'status', damit ein
# present/absent-Konflikt erkannt wird) für merge_metas().
_META_LIST_KEYS = {
    "findings": lambda x: (x.get("symptom"), x.get("attribute")),
    "disorders": lambda x: (x.get("disorder"), x.get("attribute")),
    "causal_relations": lambda x: (x.get("relation"), x.get("a"), x.get("b")),
    "reported": lambda x: (x.get("node"),),
    "durations": lambda x: (x.get("symptom"),),
    "counts": lambda x: (x.get("symptom"), x.get("window"), x.get("per")),
    "measurements": lambda x: (x.get("measurement"),),
    "temporal_relations": lambda x: tuple(sorted((k, str(v)) for k, v in x.items()
                                                 if k != "status" and k != "evidence")),
}


def merge_metas(base: dict, delta: dict) -> dict:
    """Mergt einen Refinement-Pass (delta) in ein Basis-Meta (Refinement-Schleife).

    Vereinigt die Listenfelder dedupliziert; bei present/absent-Konflikt auf dem
    gleichen Schlüssel gewinnt 'present' (der Reasoner liest present vor absent).
    'period' wird übernommen, falls Basis keines hat. Reine Daten — keine DB."""
    out = {k: v for k, v in base.items()}
    for field, keyfn in _META_LIST_KEYS.items():
        merged, by_key = [], {}
        for src in (base.get(field, []) or []), (delta.get(field, []) or []):
            for x in src:
                kk = keyfn(x)
                if kk not in by_key:
                    by_key[kk] = x
                    merged.append(x)
                else:
                    # Konflikt: present verdrängt ein zuvor eingetragenes absent.
                    prev = by_key[kk]
                    if x.get("status") == "present" and prev.get("status") != "present":
                        merged[merged.index(prev)] = x
                        by_key[kk] = x
        if merged:
            out[field] = merged
    if not isinstance(out.get("period"), dict) and isinstance(delta.get("period"), dict):
        out["period"] = delta["period"]
    return out


# =============================================================================
# 6) Worked Example + End-to-End-Selbsttest (ohne DB)
# =============================================================================
# So sähe die LLM-Ausgabe für eine "typische Aura OHNE Kopfschmerz"-Vignette aus.
EXAMPLE_META = {
    "vignette_id": "vignette_typische_aura_ohne_ks",
    "language": "de",
    "findings": [
        {"symptom": "aura", "attribute": None, "status": "present",
         "evidence": "rezidivierende Sehstörungen mit Flimmern"},
        {"symptom": "aura", "attribute": "visuell", "status": "present",
         "evidence": "Flimmerskotom im rechten Gesichtsfeld"},
        {"symptom": "aura", "attribute": "fully reversible", "status": "present",
         "evidence": "bildet sich vollständig zurück"},
        {"symptom": "aura", "attribute": "motor", "status": "absent",
         "evidence": "keine Lähmungserscheinungen"},
        {"symptom": "schmerz", "attribute": None, "status": "absent",
         "evidence": "im Anschluss kein Kopfschmerz"},
    ],
    "durations": [
        {"symptom": "aura", "min_minutes": 20, "max_minutes": 40, "comparator": "=",
         "status": "present", "evidence": "etwa 20 bis 40 Minuten"},
    ],
    "counts": [
        {"symptom": "attacke", "value": 5, "comparator": ">", "status": "present",
         "evidence": "seit Jahren immer wieder, sicher mehr als fünf Episoden"},
    ],
    "temporal_relations": [
        {"rel": "after", "from": "aura", "to": "schmerz", "status": "absent",
         "evidence": "im Anschluss kein Kopfschmerz"},
        {"rel": "overlaps", "from": "aura", "to": "schmerz", "status": "absent",
         "evidence": "kein begleitender Kopfschmerz"},
    ],
}

def run_demo() -> int:
    # 6a) Adapter prüfen
    c = meta_to_case(EXAMPLE_META)
    print("Adaptierter Case:")
    print("  present       =", sorted(c.present))
    print("  absent        =", sorted(c.absent))
    print("  attrs_present =", sorted(c.attrs_present))
    print("  attrs_absent  =", sorted(c.attrs_absent))
    print("  durations     =", c.durations)
    print("  counts        =", c.counts)
    print("  rels          =", c.rels)
    print("  rels_absent   =", c.rels_absent)

    # 6b) End-to-End gegen die Fixture: 'after' als absent -> ¬after = TRUE
    g = selftest_graph()
    memo = Reasoner(g).evaluate_case(c)
    got = memo["TEST"]
    ok = got == K.TRUE
    print(f"\nEnd-to-End (Fixture TEST): {got} -> {'OK' if ok else 'FAIL'} (erwartet TRUE)")

    # 6c) Schema-Validierung (optional, falls jsonschema installiert)
    try:
        import jsonschema
        jsonschema.validate(EXAMPLE_META, SCHEMA)
        print("Schema-Validierung: OK")
    except ImportError:
        print("Schema-Validierung: übersprungen (pip install jsonschema)")
    except Exception as e:
        print("Schema-Validierung: FAIL —", e)
        return 1
    return 0 if ok else 1

if __name__ == "__main__":
    if "--print-prompt" in sys.argv:
        print(EXTRACTION_PROMPT)
    elif "--print-schema" in sys.argv:
        print(json.dumps(SCHEMA, indent=2, ensure_ascii=False))
    else:
        raise SystemExit(run_demo())
