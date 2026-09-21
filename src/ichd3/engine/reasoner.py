#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICHD-3 Reasoner (dreiwertig, DAG-basiert)
=========================================

Lädt den geschichteten Regel-Graphen EINMAL aus Neo4j und wertet jeden Fall in
einem topologischen Bottom-up-Pass aus. Statt des binären sat-Flags + Fixpunkt
(apoc.periodic.commit) wird hier Kleene-Logik mit drei Wahrheitswerten benutzt:

    TRUE     = Kriterium erfüllt
    FALSE    = Kriterium widerlegt
    UNKNOWN  = keine Information (nicht erhoben)

Damit verschwindet die Closed-World-Falle: "nicht erfasst" wird als UNKNOWN
propagiert, nicht als FALSE. Negation (negierte GROUNDS_ON-Kanten, negierte
TEMPORAL-Knoten, NOT-Kind) ist als Kleene-NOT umgesetzt: NOT(UNKNOWN)=UNKNOWN.

Komplexität: O(V+E) pro Fall (memoisierte Tiefensuche über den DAG), kein
Iterieren über die Baumtiefe, keine DB-Schreibzugriffe.

ABHÄNGIGKEIT:  pip install neo4j
NUTZUNG:
    python ichd3_reasoner.py --selftest      # ohne DB: prüft nur die Engine
    python ichd3_reasoner.py                  # lädt aus Neo4j, rechnet Beispielfälle

VORBEHALTE (bewusst offengelegt):
  * DIFFERENTIAL ("not better accounted for") ist lokal nicht entscheidbar und
    wird als Annahme gesetzt (Default TRUE = nicht-blockierend). Siehe
    DIFFERENTIAL_ASSUMPTION. Die echte Konkurrenzauflösung zwischen Diagnosen
    ist damit NICHT gelöst.
  * Der Single-Pass setzt voraus, dass der Graph azyklisch ist. Ein Zyklus
    (z.B. wenn das Differential echte Diagnose-Querverweise zöge) wird erkannt
    und als Fehler gemeldet, nicht still falsch gerechnet.
  * Performance-Aussagen aus dem Chat sind analytisch, nicht gemessen — vor
    einer etwaigen C++-Portierung erst den Ist-Zustand benchmarken.
"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Set, Optional

# =============================================================================
# DB-Verbindung  — die Konfiguration (NEO4J_URI/USER/PASSWORD/DATABASE aus der
# Umgebung) wohnt zentral in ichd3/db.py (Neo4jConfig). Siehe main() unten.
# =============================================================================

# =============================================================================
# Kleene-Logik (dreiwertig)
# =============================================================================
class K:
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

def k_not(x: str) -> str:
    return {K.TRUE: K.FALSE, K.FALSE: K.TRUE, K.UNKNOWN: K.UNKNOWN}[x]

def k_and(xs: List[str]) -> str:
    if any(x == K.FALSE for x in xs):
        return K.FALSE
    if any(x == K.UNKNOWN for x in xs):
        return K.UNKNOWN
    return K.TRUE  # leere Liste => TRUE (vakuos)

def k_or(xs: List[str]) -> str:
    if any(x == K.TRUE for x in xs):
        return K.TRUE
    if any(x == K.UNKNOWN for x in xs):
        return K.UNKNOWN
    return K.FALSE  # leere Liste => FALSE

def k_kofn(xs: List[str], k: int, comparator: str = ">=") -> str:
    """Polythetisch '<Anzahl erfüllter Kinder> comparator k' — dreiwertig.

    Die wahre Trefferzahl liegt im Intervall [lo, hi]: lo = sicher erfüllte
    (TRUE), hi = lo + Unbekannte (UNKNOWN optimistisch als erfüllt). FALSE/TRUE
    nur, wenn das Intervall die Schwelle eindeutig trennt, sonst UNKNOWN.

      comparator '>=' (Default): mind. k     — TRUE: lo>=k ;       FALSE: hi<k
      comparator '<='          : höchstens k  — TRUE: hi<=k ;       FALSE: lo>k
      comparator '='           : genau k      — TRUE: lo==k==hi ;   FALSE: k<lo oder k>hi
    """
    t = sum(1 for x in xs if x == K.TRUE)
    f = sum(1 for x in xs if x == K.FALSE)
    lo = t                 # sicher erfüllte Trefferzahl (Untergrenze)
    hi = len(xs) - f       # erreichbares Maximum (Unbekannte optimistisch)
    comp = comparator or ">="
    if comp == "<=":
        if hi <= k:
            return K.TRUE
        if lo > k:
            return K.FALSE
        return K.UNKNOWN
    if comp == "=":
        if lo == k and hi == k:
            return K.TRUE
        if k < lo or k > hi:
            return K.FALSE
        return K.UNKNOWN
    # ">=" (Default)
    if lo >= k:
        return K.TRUE
    if hi < k:             # erreichbares Maximum < k
        return K.FALSE
    return K.UNKNOWN

# Annahme für das nicht-lokal entscheidbare Differentialkriterium.
# TRUE = nicht-blockierend (eine Diagnose kann RELEVANT werden).
# Auf K.UNKNOWN setzen, wenn das Differential die Diagnose bewusst offen halten soll.
DIFFERENTIAL_ASSUMPTION = K.TRUE

# REPORTED-Knoten sind reine Freitext-Befunde (note) — strukturell nicht
# verifizierbar. Default UNKNOWN (ehrlich: wir wissen es nicht). Auf K.TRUE
# setzen, um anamnestisch berichtete Tatsachen als gegeben anzunehmen.
REPORTED_ASSUMPTION = K.UNKNOWN


@dataclass(frozen=True)
class EvalPolicy:
    """Die drei Stellen, an denen der Graph *bewusst* nicht lokal entscheidbar
    ist. Sie waren bisher Modulkonstanten — ein Sensitivitätslauf zwang dazu,
    die Datei zu editieren. Als Objekt sind sie pro Auswertung wählbar, und die
    Annahme steht im Ergebnis statt im Quelltext.

    Die Defaults reproduzieren das bisherige Verhalten bitgenau.

    * ``reported`` — REPORTED-Knoten sind Freitext-Befunde ohne Struktur.
      ``case``: eine explizite Adjudikation im Fall gilt, sonst UNKNOWN (Default).
      ``unknown``: streng, Adjudikation wird ignoriert.
      ``true``: nachgiebig, anamnestisch Berichtetes gilt als gegeben.

    * ``differential`` — „not better accounted for by another ICHD-3 diagnosis".
      ``true``: nicht-blockierender Platzhalter (Default), ``unknown``: zeigt
      seinen Diskriminanz-Beitrag (die Diagnose bleibt bewusst offen).

    * ``external_diagnosis`` — ein Verweis auf eine Diagnose ohne SATISFIED_BY
      (im Graphen nur Überschrift, z.B. 8.1) ist lokal nicht auswertbar.
      ``unknown`` (Default) oder ``true`` (Verweis als erfüllt annehmen).

    Namen und Semantik spiegeln ``backend/case_eval.py::EvalPolicy`` im
    annotations_tool, damit Verdikte beider Evaluatoren vergleichbar bleiben.
    """

    reported: str = "case"
    differential: str = "true"
    external_diagnosis: str = "unknown"
    # #4 (AP-G3): was das SCHWEIGEN einer Vignette über ein Merkmal bedeutet.
    # 'unknown' (Default) = heutiges Verhalten (nicht erwähnt → UNKNOWN). 'graph' =
    # ein nicht erwähntes `closed_world`-Merkmal gilt als abwesend (→ FALSE),
    # `open_world` bleibt UNKNOWN. Riskanter Ranking-Hebel → bewusst opt-in.
    epistemic: str = "unknown"
    # #21 (R9): WO der Ausschluss (MUTUALLY_EXCLUSIVE) greift.
    # 'taxonomy' (Default) = heutiges Verhalten: nur ueber `_child_status` beim
    # Aufloesen eines Taxonomie-ELTERN-Knotens. Ein direkt referenziertes BLATT-
    # Attribut sah den Ausschluss nie — `resolve_attr` kehrt vor der Abfrage zurueck,
    # sobald `taxonomy` keinen Eintrag hat. Die 6 deklarierten Paare waren dadurch
    # groesstenteils inert (nur die Intensitaets-Achse hat Taxonomie-Eltern).
    # 'all' = der Ausschluss gilt auch fuer Blaetter: ein wahres, wechselseitig
    # ausschliessendes Geschwister macht das Attribut FALSE. Opt-in, weil es
    # zusaetzliche FALSE-Verdikte erzeugt (Ranking-Hebel, s. graph-adaptation-findings §6).
    exclusive: str = "taxonomy"
    # Alias-Aufloesung (#23): das Vokabular fuehrt denselben Begriff mehrfach
    # (`unilateral` und `unilateral_location` stehen im Graphen unter EINEM OR).
    # 'off' (Default) = heutiges Verhalten: ein Fall, der die eine Id behauptet,
    # laesst die andere UNKNOWN. 'graph' = Alias-Geschwister werden mitgelesen.
    aliases: str = "off"

    def __post_init__(self) -> None:
        if self.reported not in ("case", "unknown", "true"):
            raise ValueError(f"EvalPolicy.reported: {self.reported!r}")
        if self.differential not in ("true", "unknown"):
            raise ValueError(f"EvalPolicy.differential: {self.differential!r}")
        if self.external_diagnosis not in ("unknown", "true"):
            raise ValueError(f"EvalPolicy.external_diagnosis: {self.external_diagnosis!r}")
        if self.epistemic not in ("unknown", "graph"):
            raise ValueError(f"EvalPolicy.epistemic: {self.epistemic!r}")
        if self.exclusive not in ("taxonomy", "all"):
            raise ValueError(f"EvalPolicy.exclusive: {self.exclusive!r}")
        if self.aliases not in ("off", "graph"):
            raise ValueError(f"EvalPolicy.aliases: {self.aliases!r}")

    # -- Kleene-Werte der drei Annahmen ---------------------------------------
    def differential_value(self) -> str:
        return K.TRUE if self.differential == "true" else K.UNKNOWN

    def external_value(self) -> str:
        return K.TRUE if self.external_diagnosis == "true" else K.UNKNOWN

    def reported_value(self, nid: str, case: "Case") -> str:
        if self.reported == "true":
            return K.TRUE
        if self.reported == "unknown":
            return K.UNKNOWN
        if nid in case.reported_present:
            return K.TRUE
        if nid in case.reported_absent:
            return K.FALSE
        return K.UNKNOWN


DEFAULT_POLICY = EvalPolicy()

# Zeiteinheiten -> Minuten (für CAUSAL_TEMPORAL-Schwellen).
_UNIT_MIN = {"sec": 1 / 60, "second": 1 / 60, "seconds": 1 / 60, "s": 1 / 60,
             "minute": 1, "minutes": 1, "min": 1, "hour": 60, "hours": 60,
             "h": 60, "day": 1440, "days": 1440,
             "week": 10080, "weeks": 10080, "month": 43200, "months": 43200,
             "year": 525600, "years": 525600}

# =============================================================================
# Falldaten — present/absent/unknown für jede Faktenart
# =============================================================================
def interval_verdict(lo: float, hi: Optional[float], comp: Optional[str], nmin, nmax) -> str:
    """Dreiwertiger Vergleich eines Wert-INTERVALLS [lo, hi] (hi None = offen) gegen eine
    QUANTIFIER-Bedingung: TRUE, wenn jeder Wert im Intervall sie erfuellt, FALSE, wenn keiner,
    sonst UNKNOWN. Fuer einen Punktwert (lo == hi) identisch mit dem bisherigen Vergleich."""
    h = float("inf") if hi is None else hi
    if comp == "<=":
        if nmax is None:
            return K.TRUE
        return K.TRUE if h <= nmax else K.FALSE if lo > nmax else K.UNKNOWN
    if comp == "<":
        if nmax is None:
            return K.TRUE
        return K.TRUE if h < nmax else K.FALSE if lo >= nmax else K.UNKNOWN
    if comp == ">":
        t = nmin if nmin is not None else 0
        return K.TRUE if lo > t else K.FALSE if h <= t else K.UNKNOWN
    t = nmin or 1                                 # None / ">=" : Mindestwert
    return K.TRUE if lo >= t else K.FALSE if h < t else K.UNKNOWN


@dataclass
class Case:
    name: str = "unbenannt"
    present: Set[str] = field(default_factory=set)            # Symptome sicher vorhanden
    absent: Set[str] = field(default_factory=set)             # Symptome sicher abwesend
    attrs_present: Set[Tuple[str, str]] = field(default_factory=set)  # (symptom, attribut) vorhanden
    attrs_absent: Set[Tuple[str, str]] = field(default_factory=set)   # (symptom, attribut) abwesend
    durations: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # sym -> (min,max) gemessen
    counts: Dict[str, int] = field(default_factory=dict)      # sym -> KUMULATIVE Anzahl (lifetime)
    measurements: Dict[str, float] = field(default_factory=dict)  # entity -> Messwert (comparison-GROUND)
    # Raten pro Zeitfenster (per_period-QUANTIFIER): (scope, periode) -> Wert,
    # z.B. ('kopfschmerztag','month')->15, ('attacke','day')->2. Bewusst GETRENNT
    # vom lifetime-Zähler: eine Tagesfrequenz ist KEINE Lebenszeit-Gesamtzahl.
    rates: Dict[Tuple[str, Optional[str]], float] = field(default_factory=dict)
    rels: List[dict] = field(default_factory=list)            # Relationen, die GELTEN
    rels_absent: List[dict] = field(default_factory=list)     # Relationen, die sicher NICHT gelten
    # Ursächliche Erkrankungen/Substanzen (Sekundärkopfschmerz): Disorder-Achse.
    disorders_present: Set[str] = field(default_factory=set)  # Disorder sicher vorhanden
    disorders_absent: Set[str] = field(default_factory=set)   # Disorder sicher abwesend
    # Achsen-Ground (Flag, Default AUS): Symptom-Typ-Entities, deren Symptom-Präsenz einen
    # Disorder-Ground erfüllen darf (disorder<-symptom-Fix, s. apply_axis_ground). Wirkt NUR
    # über disorder(); der Roh-Set disorders_present bleibt unangetastet (damit exclusion/
    # competing-Logik unverändert). Leer = kein Verhaltensunterschied.
    axis_ground_syms: Set[str] = field(default_factory=set)
    # Kausale Kopplungen (Kausalitäts-Nachweis): je {relation, ents(frozenset),
    # within_minutes|None}. causals = gelten, causals_absent = gelten sicher nicht.
    causals: List[dict] = field(default_factory=list)
    causals_absent: List[dict] = field(default_factory=list)
    # Adjudizierte REPORTED-Befunde (Freitext-Notes), je REPORTED-Knoten-id.
    reported_present: Set[str] = field(default_factory=set)
    reported_absent: Set[str] = field(default_factory=set)
    # Verlaufsmuster des Kopfschmerzes (episodisch in Bouts vs. chronisch).
    # Schlüssel: mode, active_period_days, sustained_months, longest_remission_months.
    period: Dict[str, Any] = field(default_factory=dict)
    # Zahl mit Vergleichsoperator ("unter 1 Tag/Monat", "mehr als 5 Attacken"): ein
    # INTERVALL (lo, hi) statt eines Punktwerts, hi None = nach oben offen; offene Grenzen
    # sind per EPS eingerechnet (patient_model.meta_to_case). Getrennt von counts/rates:
    # ein Punktwert behaelt Vorrang, und Fuzzy/CF (lesen nur Punktwerte) bleiben neutral.
    count_bounds: Dict[str, Tuple[float, Optional[float]]] = field(default_factory=dict)
    rate_bounds: Dict[Tuple[str, Optional[str]], Tuple[float, Optional[float]]] = field(default_factory=dict)

    # -- atomare Kleene-Auswertungen ------------------------------------------
    def sym(self, s: str) -> str:
        if s in self.present: return K.TRUE
        if s in self.absent:  return K.FALSE
        return K.UNKNOWN

    def disorder(self, d: str) -> str:
        if d in self.disorders_present: return K.TRUE
        if d in self.disorders_absent:  return K.FALSE
        if d in self.axis_ground_syms:
            if d in self.present: return K.TRUE
            if d in self.absent:  return K.FALSE
        return K.UNKNOWN

    def attr(self, s: str, a: str) -> str:
        if (s, a) in self.attrs_present: return K.TRUE
        if (s, a) in self.attrs_absent:  return K.FALSE
        return K.UNKNOWN

    def duration(self, s: str, nmin, nmax, bounds: str = "[]") -> str:
        """Beobachtetes Dauer-Intervall gegen das geforderte prüfen.

        Dreiwertig nach Lage der Intervalle: beobachtet ⊆ gefordert -> TRUE;
        beide DISJUNKT -> FALSE; TEILÜBERLAPPUNG -> UNKNOWN. Letzteres ist der
        Kleene-Kern: beobachtet 2–100 min gegen gefordert 4–72 min heißt, dass
        manche Attacken passen und manche nicht — der Fall entscheidet die Frage
        nicht, er widerlegt sie auch nicht. Vorher lieferte jeder Fall, der nicht
        vollständig im Fenster lag, FALSE (90 falsche Widerlegungen im Gold-Korpus).
        """
        if s not in self.durations:
            return K.UNKNOWN
        lo, hi = self.durations[s]
        # Fehlende Fallgrenze, gegen die geprüft werden müsste -> nicht
        # entscheidbar (Kleene: UNKNOWN statt Absturz/Falschwert).
        if nmin is not None and lo is None:
            return K.UNKNOWN
        if hi is None and nmax is not None:
            # Nach oben offenes Intervall ("seit über 72 Stunden"): nur die Untergrenze
            # entscheidet — liegt sie schon über dem Maximum, ist das Fenster verfehlt.
            upper_open = (bounds[1:2] or "]") == ")"
            above = lo is not None and (lo >= nmax if upper_open else lo > nmax)
            return K.FALSE if above else K.UNKNOWN
        # Intervallgrenzen (§4): '[' geschlossen (>=/<=), '(' bzw. ')' offen (>/<).
        # Erstes Zeichen = untere, zweites = obere Grenze; Default geschlossen.
        bl = bounds[0] if bounds else "["
        bu = bounds[1] if bounds and len(bounds) > 1 else "]"
        lo_ok = nmin is None or (lo > nmin if bl == "(" else lo >= nmin)
        hi_ok = nmax is None or (hi < nmax if bu == ")" else hi <= nmax)
        if lo_ok and hi_ok:                       # beobachtet ⊆ gefordert
            return K.TRUE
        # Disjunkt: das beobachtete Intervall liegt ganz unter- bzw. oberhalb.
        below = nmin is not None and hi is not None and (hi <= nmin if bl == "(" else hi < nmin)
        above = nmax is not None and (lo >= nmax if bu == ")" else lo > nmax)
        return K.FALSE if (below or above) else K.UNKNOWN

    def count(self, s: str, nmin) -> str:
        if s in self.counts:
            return K.TRUE if self.counts[s] >= (nmin or 1) else K.FALSE
        if s in self.count_bounds:
            return interval_verdict(*self.count_bounds[s], ">=", nmin, None)
        return K.UNKNOWN

    def measure(self, ent: str, comparator: str, value) -> str:
        """Numerischer Vergleich eines Messwerts (comparison-GROUND, §3.2). Fehlt
        der Messwert -> UNKNOWN (kein Präsenz-Kurzschluss zu fälschlich TRUE)."""
        v = self.measurements.get(ent)
        if v is None or value is None:
            return K.UNKNOWN
        ok = {">=": v >= value, ">": v > value, "<=": v <= value, "<": v < value,
              "=": v == value, "!=": v != value}.get(comparator)
        if ok is None:
            return K.UNKNOWN
        return K.TRUE if ok else K.FALSE

    def quant(self, scope: str, props: dict) -> str:
        """Fenster-bewusste QUANTIFIER-Bedingung. Trennt sauber:
          * windowKind 'lifetime'   -> kumulativer Zähler (counts[scope]),
          * windowKind 'per_period' -> Rate pro windowPeriod (rates[(scope,per)]).
        Vergleich per 'comparator': '<=' gegen 'max', sonst '>=' gegen 'min'.
        Fehlt der passende Wert -> UNKNOWN (fail-safe), NIE FALSE aus dem falschen
        Fenster: eine Tagesfrequenz darf einen Lebenszeit-Mindestwert weder
        erfüllen noch widerlegen."""
        comp = props.get("comparator", ">=")
        nmin, nmax = props.get("min"), props.get("max")
        if props.get("windowKind") == "per_period":
            per = props.get("windowPeriod")
            val = self.rates.get((scope, per))
            if val is None:                       # periodenlose Rate als Rückfall
                val = self.rates.get((scope, None))
        else:                                     # lifetime (Default)
            val = self.counts.get(scope)
        if val is not None:
            verdict = interval_verdict(val, val, comp, nmin, nmax)
        else:
            # Zahl mit Vergleichsoperator: Intervall statt Punktwert, gleiche Fensterwahl.
            if props.get("windowKind") == "per_period":
                b = self.rate_bounds.get((scope, per))
                if b is None:
                    b = self.rate_bounds.get((scope, None))
            else:
                b = self.count_bounds.get(scope)
            if b is None:
                return K.UNKNOWN
            verdict = interval_verdict(b[0], b[1], comp, nmin, nmax)
        if verdict == K.FALSE:
            return K.FALSE
        # Zusatzforderung 'sustained' (z.B. ">=15 Tage/Monat über >=3 Monate"):
        # die Rate muss über sustainedValue (in Monate umgerechnet) gehalten sein.
        sv = props.get("sustainedValue")
        if sv is not None:
            need = sv * {"month": 1, "months": 1, "week": 0.25, "weeks": 0.25,
                         "day": 1 / 30, "days": 1 / 30, "year": 12, "years": 12
                         }.get((props.get("sustainedUnit") or "months").lower(), 1)
            have = self.period.get("sustained_months")
            if have is None:
                return K.UNKNOWN                  # gefordert, aber Verlaufsdauer unbekannt
            if have < need:
                return K.FALSE
        return verdict

    def relational(self, rel: str, g: Set[str], node_min, node_max) -> str:
        """Wahrheit der Proposition '<rel> zwischen den gegroundeten Symptomen,
        Schwelle erfüllt'. TRUE/FALSE/UNKNOWN.

        `meets` schwellt GENERISCH auf node_min/node_max — der Graph liefert die
        Fenstergrenzen auf JEDER fenstertragenden Relation (AP-G1), also nicht mehr
        über eine rel-String-Whitelist. Damit greift die Schranke auch für die
        Katalog-Tokens `within`/`before` (Aura-Timing ≤ 60 min, 13.8 ≤ 336 h), und
        fensterlose Relationen (`during`/`overlaps`/`concurrent`) fallen korrekt auf
        Existenz. Dreiwertig aggregiert wie `causal`: eine unbezifferte Latenz bei
        geforderter Schranke ist kein Gegenbeweis (UNKNOWN), nicht Auto-Pass — der
        alte `within`-Default 0 machte jede fensterlose `after`-Angabe TRUE."""
        def matches(f) -> bool:
            if f.get("rel") != rel:
                return False
            if rel in ("succession", "gradual_spread"):
                return f.get("sym") in g
            # within/before/during: die Extraktion verankert diese Katalog-Relationen
            # überwiegend an EINEM Symptom (`sym`), obwohl der Knoten beide Enden
            # groundet (z.B. Aura ∧ headache). Fehlt die from/to-Paarung, den sym-Anker
            # akzeptieren — wie succession/gradual_spread. Sonst blieben ~130 within-,
            # ~78 during-Fakten still unmatched (UNKNOWN). Die echten from/to-Fakten
            # laufen weiter über die Paar-Prüfung.
            if rel in ("within", "before", "during") \
                    and f.get("from") is None and f.get("to") is None:
                return f.get("sym") in g
            # übrige paarweise Relationen (after/overlaps/concurrent)
            return f.get("from") in g and f.get("to") in g

        def meets(f) -> Optional[bool]:
            if rel == "succession":
                return node_min is None or f.get("count", 0) >= node_min
            if rel == "gradual_spread":
                return node_min is None or f.get("min", 0) >= node_min
            if node_min is None and node_max is None:
                return True                       # fensterlos → Existenz genügt
            w = f.get("within")
            if w is None:
                return None                       # Schranke gefordert, Fall schweigt → UNKNOWN
            return ((node_min is None or w >= node_min)
                    and (node_max is None or w <= node_max))

        pos = [f for f in self.rels if matches(f)]
        if pos:
            vs = [meets(f) for f in pos]
            if any(v is True for v in vs):
                return K.TRUE
            if any(v is None for v in vs):        # könnte die Schranke noch halten
                return K.UNKNOWN
            return K.FALSE                        # alle bekannt, alle Schranke verfehlt
        if any(matches(f) for f in self.rels_absent):
            return K.FALSE
        return K.UNKNOWN

    def causal(self, relation: str, ents: Set[str], within_max=None,
               within_min=None) -> str:
        """Wahrheit einer kausalen Kopplung '<relation> zwischen den gegroundeten
        Entitäten' (CAUSAL_COUPLING / CAUSAL_TEMPORAL). Entitäten werden als Menge
        gematcht (Reihenfolge egal); bei Zeitschranke muss die Latenz im Fenster
        [within_min, within_max] liegen (#2: BEIDE Grenzen, nicht nur die obere —
        'zwischen 7 Tagen und 3 Monaten danach' unterscheidet die verzögerte
        Appendix-Variante von der akuten '≤7 Tage'-Diagnose).

        Kleene-Kern: eine UNBEZIFFERTE Latenz ist kein Gegenbeweis. Sagt der Fall
        'der Kopfschmerz entwickelte sich nach der Störung', nennt aber keine
        Zeitspanne, während der Graph ein Fenster fordert, ist die Schranke
        UNBEKANNT — nicht verletzt. Nur eine bezifferte Latenz außerhalb des
        Fensters widerlegt (FALSE). Die Verwechslung machte den Sekundärkopfschmerz
        hart unerfüllbar (Sekundär→Primär-Kollaps): 238 Gold-Ziele wurden
        ausgeschlossen, obwohl die Vignette die Kopplung behauptet.

        Mehrere passende Fakten werden existenziell gelesen: genügt EINER dem
        Fenster, ist die Kopplung TRUE; sonst kippt schon EIN Fakt mit unbekannter
        Latenz das Ergebnis auf UNKNOWN (er könnte es noch halten)."""
        want = frozenset(ents)

        def matches(f) -> bool:
            return f.get("relation") == relation and f.get("ents") == want

        def in_window(w) -> bool:
            return ((within_min is None or w >= within_min)
                    and (within_max is None or w <= within_max))

        pos = [f for f in self.causals if matches(f)]
        if pos:
            if within_min is None and within_max is None:   # keine Schranke gefordert
                return K.TRUE
            ws = [f.get("within_minutes") for f in pos]
            if any(w is not None and in_window(w) for w in ws):
                return K.TRUE                    # eine belegte Latenz hält das Fenster
            if any(w is None for w in ws):
                return K.UNKNOWN                 # Kopplung belegt, Latenz unbeziffert
            return K.FALSE                       # alle Latenzen bekannt, alle außerhalb
        if any(matches(f) for f in self.causals_absent):
            return K.FALSE
        return K.UNKNOWN

    def period_value(self, props: dict) -> str:
        """Bewertet einen PERIOD_PATTERN-Knoten gegen das Verlaufsmuster des Falls.
        episodic: aktive Phase (Bout) in [activeMin, activeMax]; chronic:
        Dauer >= sustainedMonths und längste Remission <= remissionMaxMonths."""
        p = self.period
        if not p or not p.get("mode"):
            return K.UNKNOWN
        node_mode = props.get("mode")
        if node_mode and p.get("mode") != node_mode:
            return K.FALSE                       # episodisch vs. chronisch widerlegt

        if node_mode == "episodic":
            amin, amax = props.get("activeMin"), props.get("activeMax")
            if amin is None and amax is None:
                return K.TRUE                    # Modus allein definiert den Knoten
            d = p.get("active_period_days")
            if d is None:
                return K.UNKNOWN
            f = {"days": 1, "day": 1, "weeks": 7, "months": 30,
                 "years": 365}.get((props.get("activeUnit") or "days").lower(), 1)
            lo = amin * f if amin is not None else None
            hi = amax * f if amax is not None else None
            ok = (lo is None or d >= lo) and (hi is None or d <= hi)
            return K.TRUE if ok else K.FALSE

        if node_mode == "chronic":
            sm_need, rm_max = props.get("sustainedMonths"), props.get("remissionMaxMonths")
            sm, rm = p.get("sustained_months"), p.get("longest_remission_months")
            checks = []
            if sm_need is not None:
                if sm is None:
                    return K.UNKNOWN
                checks.append(sm >= sm_need)
            if rm_max is not None:
                if rm is None:
                    return K.UNKNOWN
                checks.append(rm <= rm_max)
            return K.TRUE if all(checks) else (K.FALSE if checks else K.TRUE)

        return K.UNKNOWN

# =============================================================================
# Regel-Graph (Quelle-agnostisch: aus Neo4j ODER aus dict)
# =============================================================================
@dataclass
class RuleGraph:
    kind: Dict[str, Optional[str]]        # logic-id -> kind ; entity-id -> None
    labels: Dict[str, Set[str]]           # node-id -> labels (ohne 'Ontology')
    props: Dict[str, dict]                # logic-id -> properties
    operands: Dict[str, List[str]]        # logic-id -> [entity-id...]
    grounds: Dict[str, List[Tuple[str, str, bool]]]  # logic-id -> [(target,label,negated)]
    satisfied_by: Dict[str, str]          # entity-id -> logic-id
    # OPERAND-Kantenproperties, (src,dst) -> props. Trägt u.a. 'missing' der
    # relaxed_criteria_reference ("erfüllt alle bis auf N Kriterien von X").
    operand_props: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    # AP-G7 Ontologie-Axiome (Diskriminanz) + AP-G3 Epistemik (Schweigen-Semantik).
    # Additiv, Leer-Defaults → Alt-Bundles ohne diese Felder verhalten sich wie bisher.
    taxonomy: Dict[str, List[str]] = field(default_factory=dict)    # IS_A: parent -> [children]
    exhaustive: Set[str] = field(default_factory=set)              # Eltern mit vollständiger Kinderliste
    exclusive: Dict[str, Set[str]] = field(default_factory=dict)   # MUTUALLY_EXCLUSIVE: id -> {ids}
    epistemic: Dict[str, str] = field(default_factory=dict)        # entity -> 'closed_world'|'open_world'
    # #23: SAME_AS. MUSS ans Ende — die uebrigen Felder werden von `from_neo4j`/`from_dict`
    # POSITIONELL uebergeben; ein Einschub davor verschiebt still epistemic in dieses Feld.
    aliases: Dict[str, Set[str]] = field(default_factory=dict)     # id -> {gleichbedeutende ids}
    # #24: Negations-Knoten `a ≡ ¬(b1 ∨ … ∨ bn)` und erschoepfende Achsen (Value Partitions,
    # Mitglieder = Alias-Gruppen). Beides NUR aus dem Sidecar (axioms.merge_axioms), nicht
    # serialisiert — der Graph-Fingerprint bleibt davon unberuehrt.
    complement: Dict[str, Set[str]] = field(default_factory=dict)
    partitions: List[List[List[str]]] = field(default_factory=list)

    def is_logic(self, nid: str) -> bool:
        return self.kind.get(nid) is not None or nid in self.props

    @classmethod
    def from_neo4j(cls, driver, database: str) -> "RuleGraph":
        kind, labels, props = {}, {}, {}
        operands: Dict[str, List[str]] = {}
        operand_props: Dict[Tuple[str, str], dict] = {}
        grounds: Dict[str, List[Tuple[str, str, bool]]] = {}
        satisfied_by: Dict[str, str] = {}
        with driver.session(database=database) as s:
            for r in s.run("MATCH (n:Logic) RETURN n.id AS id, n.kind AS kind, properties(n) AS props"):
                kind[r["id"]] = r["kind"]
                props[r["id"]] = dict(r["props"])
                labels[r["id"]] = {"Logic"}
            for r in s.run("MATCH (n) WHERE n:Diagnosis OR n:Criterion OR n:SubCriterion "
                           "RETURN n.id AS id, [l IN labels(n) WHERE l<>'Ontology'] AS labels, "
                           "n.name AS name, n.label AS label"):
                kind[r["id"]] = None
                labels[r["id"]] = set(r["labels"])
                props[r["id"]] = {"name": r["name"]}
                # Buchstabenkriterien (:Criterion.label = 'A'..'D') für die
                # nodes-Verweissemantik; nur additiv, alte Exporte bleiben gültig.
                if r["label"] is not None:
                    props[r["id"]]["label"] = r["label"]
            for r in s.run("MATCH (a:Logic)-[e:OPERAND]->(b) "
                           "RETURN a.id AS src, b.id AS dst, properties(e) AS props"):
                operands.setdefault(r["src"], []).append(r["dst"])
                if r["props"]:
                    operand_props[(r["src"], r["dst"])] = dict(r["props"])
            for r in s.run("MATCH (a:Logic)-[r:GROUNDS_ON]->(b) "
                           "RETURN a.id AS src, b.id AS dst, "
                           "[l IN labels(b) WHERE l<>'Ontology'][0] AS tlabel, "
                           "coalesce(r.negated,false) AS negated"):
                grounds.setdefault(r["src"], []).append((r["dst"], r["tlabel"], bool(r["negated"])))
            for r in s.run("MATCH (a)-[:SATISFIED_BY]->(b:Logic) "
                           "WHERE a:Diagnosis OR a:Criterion OR a:SubCriterion "
                           "RETURN a.id AS src, b.id AS dst"):
                satisfied_by[r["src"]] = r["dst"]
            # AP-G7: Ontologie-Axiome (Subsumption/Abschluss/Exklusivität).
            taxonomy: Dict[str, List[str]] = {}
            exhaustive: Set[str] = set()
            exclusive: Dict[str, Set[str]] = {}
            for r in s.run("MATCH (a)-[:IS_A]->(b) RETURN a.id AS child, b.id AS parent"):
                taxonomy.setdefault(r["parent"], []).append(r["child"])
            for r in s.run("MATCH (n) WHERE n.exhaustive RETURN n.id AS id"):
                exhaustive.add(r["id"])
            for r in s.run("MATCH (a)-[:MUTUALLY_EXCLUSIVE]-(b) RETURN a.id AS a, b.id AS b"):
                exclusive.setdefault(r["a"], set()).add(r["b"])
            # AP-G3: Epistemik (open/closed-world je Ontologie-Knoten).
            epistemic: Dict[str, str] = {}
            for r in s.run("MATCH (n) WHERE n.epistemic IS NOT NULL "
                           "RETURN n.id AS id, n.epistemic AS e"):
                epistemic[r["id"]] = r["e"]
        return cls(kind, labels, props, operands, grounds, satisfied_by, operand_props,
                   taxonomy, exhaustive, exclusive, epistemic)

    def to_dict(self) -> dict:
        """JSON-serialisierbarer Gegenpart zu `from_dict`: Sets→Listen,
        `operand_props`-Tupelschlüssel→"src|dst". Erlaubt, den aus Neo4j
        geladenen Graphen einmal zu exportieren und spätere Reasoner-Läufe rein
        offline via `from_dict` zu fahren (kein DB-Container zur Laufzeit)."""
        d = {
            "kind": dict(self.kind),
            "labels": {k: sorted(v) for k, v in self.labels.items()},
            "props": dict(self.props),
            "operands": {k: list(v) for k, v in self.operands.items()},
            "grounds": {k: [list(t) for t in v] for k, v in self.grounds.items()},
            "satisfied_by": dict(self.satisfied_by),
            "operand_props": {f"{s}|{d}": v for (s, d), v in self.operand_props.items()},
        }
        # AP-G7/G3-Felder nur emittieren, wenn belegt — so bleibt der Fingerprint
        # (sha über to_dict) von Alt-Bundles ohne diese Daten unverändert.
        if self.taxonomy:
            d["taxonomy"] = {k: list(v) for k, v in self.taxonomy.items()}
        if self.exhaustive:
            d["exhaustive"] = sorted(self.exhaustive)
        if self.exclusive:
            d["exclusive"] = {k: sorted(v) for k, v in self.exclusive.items()}
        if self.epistemic:
            d["epistemic"] = dict(self.epistemic)
        if self.aliases:
            d["aliases"] = {k: sorted(v) for k, v in self.aliases.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RuleGraph":
        # operand_props optional als {"src|dst": {...}} oder {(src,dst): {...}}.
        raw = d.get("operand_props", {})
        op_props = {tuple(k.split("|", 1)) if isinstance(k, str) else k: v
                    for k, v in raw.items()}
        # Aus JSON kommen labels als Listen und grounds als Listen-of-Listen
        # zurück — auf die internen Typen (Set / Tupel) normalisieren, damit ein
        # via to_dict exportierter Graph identisch zum from_neo4j-Graphen ist.
        labels = {k: set(v) for k, v in d["labels"].items()}
        grounds = {k: [tuple(t) for t in v] for k, v in d.get("grounds", {}).items()}
        # AP-G7/G3-Felder optional (Alt-Bundles ohne sie → leer → altes Verhalten).
        taxonomy = {k: list(v) for k, v in d.get("taxonomy", {}).items()}
        exhaustive = set(d.get("exhaustive", []))
        exclusive = {k: set(v) for k, v in d.get("exclusive", {}).items()}
        epistemic = dict(d.get("epistemic", {}))
        aliases = {k: set(v) for k, v in d.get("aliases", {}).items()}
        return cls(d["kind"], labels, d.get("props", {}),
                   d.get("operands", {}), grounds, d["satisfied_by"],
                   op_props, taxonomy, exhaustive, exclusive, epistemic, aliases)

# =============================================================================
# Träger eines Zeitkriteriums (von Kleene- UND Fuzzy-Reasoner genutzt)
# =============================================================================
def temporal_carriers(g: RuleGraph, nid: str) -> List[str]:
    """Entitäten, an denen ein TEMPORAL-Knoten sein Zeitfenster misst — über
    BEIDE Achsen, Symptom vor Disorder.

    Der Decomp-Graph erdet Dauern überwiegend auf `headache` als :Disorder; ein
    Symptom-only-Filter macht 57 der 94 TEMPORAL-Knoten bedingungslos UNKNOWN.
    Die Achse ist hier Notation, keine Semantik: die Fall-Adapter spiegeln die
    Dauer über die Trägersynonyme (headache/attack/episode).

    Reihenfolge aus dem Graphen, nicht aus einer Set-Iteration — die war über
    PYTHONHASHSEED lauf-instabil, sobald ein Knoten mehrere Grounds trägt."""
    edges = g.grounds.get(nid, [])
    return ([t for (t, lbl, _) in edges if lbl == "Symptom"]
            + [t for (t, lbl, _) in edges if lbl == "Disorder"])


def duration_carrier(carriers: List[str], case: Case) -> Optional[str]:
    """Den Träger wählen, für den der Fall eine Dauer gemessen hat; sonst den
    ersten — dann entscheidet die Dauer-Auswertung ehrlich UNKNOWN."""
    return next((t for t in carriers if t in case.durations),
                carriers[0] if carriers else None)


def temporal_window(props: dict) -> Tuple[Optional[float], Optional[float]]:
    """(min, max) eines TEMPORAL-Knotens nach Minuten skalieren — INLINE, wie es
    `case_eval.py` tut, statt sich auf einen externen `normalize_duration_nodes`-
    Vorlauf zu verlassen (der ist ein Footgun: 98 der 104 TEMPORAL-Knoten stehen in
    Stunden, vergisst eine Offline-Eval den Vorlauf, prüft sie h-Fenster gegen
    min-Falldaten). IDEMPOTENT: hat `normalize_duration_nodes` schon skaliert, steht
    `unit='minute'` → Faktor 1 → kein Doppel-Skalieren. Geteilt von Kleene- und
    Fuzzy-Reasoner, damit die beiden Engines hier nicht auseinanderdriften."""
    f = _UNIT_MIN.get((props.get("unit") or "minute").lower(), 1)
    nmin, nmax = props.get("min"), props.get("max")
    return (nmin * f if nmin is not None else None,
            nmax * f if nmax is not None else None)


def causal_window(props: dict) -> Tuple[Optional[float], Optional[float]]:
    """(min, max)-Fenster einer kausalen Zeitschranke nach Minuten (#2). Liest
    `n.min`/`n.max` (AP-G1) statt des unzuverlässigen `n.within`-Alias: bei einer
    Nur-Untergrenze-Schranke (`min` gesetzt, `max=None`, z.B. 6.1.1.2 '≥3 Monate')
    trägt der Alias die UNTERgrenze, wurde aber als Obergrenze gelesen — Fenster
    genau falsch herum. Rückfall auf den Alias (als Obergrenze) nur, wenn ein alter
    Graph weder min noch max trägt."""
    f = _UNIT_MIN.get((props.get("unit") or "minute").lower(), 1)
    nmin, nmax = props.get("min"), props.get("max")
    if nmin is None and nmax is None and props.get("within") is not None:
        return (None, props["within"] * f)
    return (nmin * f if nmin is not None else None,
            nmax * f if nmax is not None else None)


def alias_lookup(g: "RuleGraph", lookup, ent: str, mode: str = "off") -> str:
    """Entitaet nachschlagen, unter ``mode='graph'`` auch ueber ihre Alias-Geschwister (#23).

    Das Vokabular fuehrt denselben Begriff mehrfach — `unilateral` und `unilateral_location`
    stehen im Graphen sogar unter EINEM OR, sind also nachweislich dasselbe gemeint. Ohne
    Aufloesung laesst ein Fall, der die eine Id behauptet, die andere UNKNOWN; das ist ein
    reines Recall-Leck, unabhaengig von jeder Logik (docs/reports/exclusion-vokabular-audit.md).

    TRUE hat Vorrang vor FALSE: Behauptungen sind positiv formuliert, und ein Alias-Geschwister,
    das der Fall verneint, soll eine anderswo behauptete Praesenz nicht ueberschreiben."""
    v = lookup(ent)
    if v != K.UNKNOWN or mode != "graph":
        return v
    sibs = g.aliases.get(ent)
    if not sibs:
        return v
    vals = [lookup(s) for s in sibs]
    if K.TRUE in vals:
        return K.TRUE
    return K.FALSE if K.FALSE in vals else K.UNKNOWN


def _child_status(g: "RuleGraph", case: Case, anchor: str, child: str,
                  alias_mode: str = "off") -> str:
    """Ein Kind-Attribut ist FALSE, wenn der Fall es verneint ODER ein
    wechselseitig ausschließendes Geschwister (MUTUALLY_EXCLUSIVE) wahr ist."""
    v = alias_lookup(g, lambda e: case.attr(anchor, e), child, alias_mode)
    if v != K.UNKNOWN:
        return v
    # Ausschluss ebenfalls ueber Aliase: ein wahres `unilateral` muss `bilateral_location`
    # auch dann auf FALSE setzen, wenn das Axiom auf `unilateral_location` deklariert ist.
    for sib in g.exclusive.get(child, ()):
        if alias_lookup(g, lambda e: case.attr(anchor, e), sib, alias_mode) == K.TRUE:
            return K.FALSE
    return K.UNKNOWN


def resolve_attr(g: "RuleGraph", case: Case, anchor: str, attr: str,
                 exclusive_mode: str = "taxonomy", alias_mode: str = "off") -> str:
    """Attribut am Anker UNTER den Ontologie-Axiomen (AP-G7, #3; Vorbild
    `case_eval.py:_attr_status`). Behauptet der Fall das Attribut direkt, gilt das.
    Schweigt er:
      1. **Subsumption** (IS_A): ein Kind wahr ⇒ Eltern TRUE (Fall sagt 'severe',
         Kriterium will 'moderate to severe').
      2. **Abschluss + Exklusivität**: `attr` exhaustive und JEDES Kind FALSE ⇒
         Eltern FALSE ('severe' schließt 'mild to moderate' aus). Ohne den
         Abschluss bliebe der Eltern-Knoten UNKNOWN — er ist die diskriminierende
         Hälfte. Geteilt von Kleene- und Fuzzy-GROUND, damit sie nicht driften.
      3. **Blatt-Ausschluss** (``exclusive_mode='all'``, #21): hat ``attr`` keine
         Taxonomie-Kinder, entscheidet trotzdem ein wahres, wechselseitig
         ausschließendes Geschwister (FALSE). Ohne diesen Zweig sieht ein direkt
         referenziertes Blatt-Attribut den Ausschluss NIE — die deklarierten
         MUTUALLY_EXCLUSIVE-Paare waren fast vollständig inert."""
    direct = alias_lookup(g, lambda e: case.attr(anchor, e), attr, alias_mode)
    if direct != K.UNKNOWN:
        return direct
    children = g.taxonomy.get(attr)
    if not children:
        # #21: ohne Taxonomie-Kinder blieb der Ausschluss bisher ungefragt, obwohl
        # `_child_status` ihn kennt — die deklarierten Paare waren fuer Blaetter inert.
        return (_child_status(g, case, anchor, attr, alias_mode)
                if exclusive_mode == "all" else K.UNKNOWN)
    vals = [_child_status(g, case, anchor, c, alias_mode) for c in children]
    if K.TRUE in vals:
        return K.TRUE
    if attr in g.exhaustive and all(v == K.FALSE for v in vals):
        return K.FALSE
    return K.UNKNOWN


def apply_epistemic(g: "RuleGraph", epistemic_mode: str, entity: str, v: str) -> str:
    """#4 (AP-G3): unter `epistemic_mode='graph'` gilt ein nicht erwähntes
    `closed_world`-Merkmal als abwesend (UNKNOWN→FALSE); `open_world`/unmarkiert
    bleiben UNKNOWN. Default `'unknown'` → unverändert. Geteilt von Kleene & Fuzzy."""
    if v == K.UNKNOWN and epistemic_mode == "graph" \
            and g.epistemic.get(entity) == "closed_world":
        return K.FALSE
    return v


# =============================================================================
# Reasoner — memoisierte Auswertung über den DAG
# =============================================================================
class Reasoner:
    def __init__(self, g: RuleGraph, policy: Optional[EvalPolicy] = None):
        self.g = g
        # Die drei nicht-lokal entscheidbaren Annahmen (DIFFERENTIAL, REPORTED,
        # Verweis auf eine undefinierte Diagnose). Default = bisheriges Verhalten.
        self.policy = policy or DEFAULT_POLICY
        # Logic-kinds, die diese Engine (noch) nicht auswertet. Statt hart zu
        # scheitern, behandelt _eval_node sie fail-safe als UNKNOWN (Kleene-
        # konform) und protokolliert sie hier: kind -> Anzahl betroffener Knoten.
        # So bleibt eine Auswertung gegen einen reicheren Graphen lauffähig,
        # ohne stillschweigend Falsches zu rechnen.
        self.unsupported_kinds: Dict[str, int] = {}

    def evaluate_case(self, case: Case) -> Dict[str, str]:
        memo: Dict[str, str] = {}
        stack: Set[str] = set()
        g = self.g

        def ev(nid: str) -> str:
            if nid in memo:
                return memo[nid]
            if nid in stack:
                raise RuntimeError(f"Zyklus im Regelgraph an Knoten {nid} — Single-Pass nicht anwendbar.")
            stack.add(nid)
            v = self._eval_node(nid, case, ev)
            stack.discard(nid)
            memo[nid] = v
            return v

        # alle Entitäten auswerten (Diagnosen/Kriterien)
        for nid, k in g.kind.items():
            if k is None:  # Entität
                ev(nid)
        return memo

    def _eval_node(self, nid: str, case: Case, ev) -> str:
        g = self.g
        k = g.kind.get(nid)

        # --- Entität: Wert ihres SATISFIED_BY-Logikknotens -------------------
        if k is None:
            target = g.satisfied_by.get(nid)
            if target:
                return ev(target)
            # Diagnose ohne Baum (im Graphen nur Überschrift, z.B. 8.1) ->
            # Policy. Andere Entitäten ohne SATISFIED_BY bleiben UNKNOWN.
            if "Diagnosis" in g.labels.get(nid, set()):
                return self.policy.external_value()
            return K.UNKNOWN

        props = g.props.get(nid, {})

        if k == "GROUND":
            return self._eval_ground(nid, case)

        if k == "TEMPORAL":
            return self._eval_temporal(nid, case)

        if k == "DIFFERENTIAL":
            return self.policy.differential_value()

        if k == "CAUSAL_COUPLING":
            ents = {t for (t, _lbl, _neg) in g.grounds.get(nid, [])}
            v = case.causal(props.get("relation"), ents, None)
            return k_not(v) if props.get("negated") else v   # §5.3: negated kippt die Kopplung

        if k == "CAUSAL_TEMPORAL":
            ents = {t for (t, _lbl, _neg) in g.grounds.get(nid, [])}
            lo, hi = causal_window(props)                    # #2: Unter- UND Obergrenze
            v = case.causal(props.get("relation"), ents, within_max=hi, within_min=lo)
            return k_not(v) if props.get("negated") else v   # Kleene: NOT(UNKNOWN)=UNKNOWN

        if k == "REPORTED":
            # Freitext-Note: im Modus 'case' gilt eine Adjudikation für DIESEN
            # Knoten, sonst entscheidet die Policy allein.
            return self.policy.reported_value(nid, case)

        if k == "PERIOD_PATTERN":
            return case.period_value(props)

        if k in ("COUNT", "SET"):
            # Aggregat-Forderung (quantity_requirement) bzw. Mengendefinition: es
            # gibt fallseitig (noch) keine Mengen-/Aggregatdaten -> ehrlich UNKNOWN.
            # Im Korpus ungenutzt, aber bewusst behandelt (nicht im unsupported-Log).
            return K.UNKNOWN

        ops = [self._operand_value(nid, o, ev) for o in g.operands.get(nid, [])]

        if k == "AND":
            return k_and(ops)
        if k == "OR":
            return k_or(ops)
        if k == "NOT":
            return k_not(k_and(ops))
        if k == "GUARD":
            # §8.3: definitions-zeitlich transparent (sat(guard) :- sat(item)); ohne
            # fallseitige Kontextdaten wird der Kontext als erfüllt angenommen ->
            # Wert des bewachten item. Besser als UNKNOWN (sonst jedes bewachte
            # Kriterium 'möglich').
            return k_and(ops)
        if k == "KOfN":
            # §5.3: 'k' ist neutral, der Sinn steckt im comparator ('<=' = höchstens k,
            # '=' = genau k). Default '>=' (fehlt die Property im Export).
            return k_kofn(ops, int(props.get("k", 1)), props.get("comparator") or ">=")
        if k == "QUANTIFIER":
            gsyms = [t for (t, lbl, _) in g.grounds.get(nid, []) if lbl == "Symptom"]
            scope = props.get("scope") or (gsyms[0] if gsyms else None)
            ccond = case.quant(scope, props) if scope else K.UNKNOWN
            return k_and([ccond, k_and(ops)])

        # Unbekannter kind: fail-safe UNKNOWN statt Absturz (z.B. CAUSAL_COUPLING,
        # PERIOD_PATTERN, REPORTED, CAUSAL_TEMPORAL aus dem vollen ICHD-3-Graphen,
        # für die noch keine Semantik implementiert ist). Wird protokolliert.
        self.unsupported_kinds[k] = self.unsupported_kinds.get(k, 0) + 1
        return K.UNKNOWN

    def _operand_value(self, src: str, dst: str, ev) -> str:
        """Wert eines OPERAND. Trägt die Kante auf eine :Diagnosis ein 'missing'
        (relaxed_criteria_reference, §3.7: 'erfüllt alle bis auf N Kriterien')
        und/oder 'nodes' (benannte Buchstabenkriterien des Ziels, z.B. ['B','C']),
        wird die Zieldiagnose VERWEIS-ausgewertet. Sonst normal ev(dst)."""
        ep = self.g.operand_props.get((src, dst))
        if ep and "Diagnosis" in self.g.labels.get(dst, set()) \
                and (ep.get("missing") is not None or ep.get("nodes")):
            return self._referenced_diagnosis(dst, ep.get("nodes"),
                                              int(ep.get("missing") or 0), ev)
        return ev(dst)

    def _split_reference_criteria(self, did: str):
        """Oberste Operanden der Zieldiagnose in (differential, substanziell)
        splitten. DIFFERENTIAL hängt entweder als direkter Logic-Operand unter
        der Wurzel oder als Entität, deren SATISFIED_BY-Logik DIFFERENTIAL ist."""
        root = self.g.satisfied_by.get(did)
        subs = self.g.operands.get(root, []) if root else []
        diff, crit = [], []
        for o in subs:
            sb = self.g.satisfied_by.get(o)
            is_diff = (self.g.kind.get(o) == "DIFFERENTIAL"
                       or (sb is not None and self.g.kind.get(sb) == "DIFFERENTIAL"))
            (diff if is_diff else crit).append(o)
        return root, diff, crit

    def _criterion_letter(self, nid: str) -> Optional[str]:
        """Buchstabe eines unmittelbaren Kriteriums: Entity-Prop 'label' (aus
        :Criterion.label exportiert), Fallback ID-Suffix nach '/' ('1.1/B'->'B')."""
        lbl = self.g.props.get(nid, {}).get("label")
        if lbl:
            return str(lbl)
        tail = nid.rsplit("/", 1)[-1]
        return tail if tail != nid else None

    def _referenced_diagnosis(self, did: str, nodes, missing: int, ev) -> str:
        """Verweis auf Zieldiagnose 'did' auswerten.

        'nodes' wählt die benannten unmittelbaren Buchstabenkriterien des Ziels
        (Matching über :Criterion.label bzw. ID-Suffix); ohne 'nodes' zählen alle
        substanziellen Wurzel-Kriterien. 'missing'=N lockert die Auswahl zu KOfN
        mit k = |Auswahl| - N (dreiwertig); beides kombinierbar (erst auswählen,
        dann lockern).

        DIFFERENTIAL-Operanden ('not better accounted for', Annahme-TRUE) sind
        keine zählbaren Kriterien: ohne 'nodes' bleiben sie Pflicht (AND), zählen
        aber NICHT in die Relaxation — sonst macht ein Gratis-TRUE die Relaxation
        vakuos (z.B. '1 von {A, E}' mit E=DIFFERENTIAL ist immer erfüllt). Bei
        expliziter 'nodes'-Auswahl entfallen sie ganz (der Verweis benennt genau
        die geforderten Kriterien).

        Fail-safes: Ziel ohne SATISFIED_BY (Verweis auf reine Überschrift wie
        8.1) -> ev(did) = UNKNOWN. Ein benannter, aber nicht auffindbarer
        Buchstabe zählt als UNKNOWN-Platzhalter (nie vakuos TRUE). Entartet die
        Relaxation (k < 1), wird konservativ die HARTE Diagnose gefordert."""
        root, diff, crit = self._split_reference_criteria(did)
        if root is None or not (diff or crit):
            return ev(did)
        if nodes:
            by_letter = {self._criterion_letter(o): o for o in crit}
            picked = [by_letter.get(str(n)) for n in nodes]
            vals = [self._operand_value(root, o, ev) if o is not None else K.UNKNOWN
                    for o in picked]
            k = len(vals) - missing
            if k < 1:
                return ev(did)
            return k_kofn(vals, k, ">=")
        k = len(crit) - missing
        if k < 1:
            return ev(did)
        crit_val = k_kofn([self._operand_value(root, o, ev) for o in crit], k, ">=")
        return k_and([self._operand_value(root, o, ev) for o in diff] + [crit_val])

    def _epistemic(self, entity: str, v: str) -> str:
        return apply_epistemic(self.g, self.policy.epistemic, entity, v)

    def _eval_ground(self, nid: str, case: Case) -> str:
        g = self.g
        edges = g.grounds.get(nid, [])
        props = g.props.get(nid, {})
        # comparison-GROUND (§3.2): numerischer Vergleich 'Messwert comparator value'.
        # Nur comparison-Knoten tragen comparator auf einem GROUND. Ohne value (z.B.
        # Entität-gegen-Entität) oder ohne Messwert -> UNKNOWN statt Präsenz-AND.
        if props.get("comparator") is not None:
            ent = next((t for (t, _l, _n) in edges), None)
            if ent is None:
                return K.UNKNOWN
            return case.measure(ent, props["comparator"], props.get("value"))
        # Anker (für die Zuordnung der Attribute): bevorzugt ein positives Symptom,
        # sonst eine positive Disorder (Sekundärkopfschmerz: Attribute wie
        # 'uebergebrauch' hängen an der ursächlichen Substanz, nicht an einem Symptom).
        anchor = next((t for (t, lbl, neg) in edges if lbl == "Symptom" and not neg), None)
        if anchor is None:
            anchor = next((t for (t, lbl, neg) in edges if lbl == "Disorder" and not neg), None)
        parts: List[str] = []
        for (t, lbl, neg) in edges:
            if lbl == "Symptom":
                p = alias_lookup(g, case.sym, t, self.policy.aliases)          # #23
            elif lbl == "Disorder":
                p = alias_lookup(g, case.disorder, t, self.policy.aliases)     # #23
            else:  # Attribute (am Symptom- oder Disorder-Anker)
                p = (resolve_attr(g, case, anchor, t, self.policy.exclusive,   # #3, #21, #23
                                   self.policy.aliases) if anchor else K.UNKNOWN)
            p = self._epistemic(t, p)                         # #4 (gated)
            parts.append(k_not(p) if neg else p)
        return k_and(parts)

    def _eval_temporal(self, nid: str, case: Case) -> str:
        """Zeitkriterium (Dauer/Relation) auswerten.

        Der Träger des Zeitfensters wird über BEIDE Achsen gesucht: der Decomp-
        Graph erdet Dauern überwiegend auf der Disorder-Achse (`headache` als
        :Disorder), 57 der 94 TEMPORAL-Knoten tragen GAR KEINEN Symptom-Ground.
        Ein Symptom-only-Filter lieferte für sie bedingungslos UNKNOWN — 61 % der
        Zeitkriterien waren tot. Die Fall-Adapter spiegeln die Dauer ohnehin über
        die Trägersynonyme (headache/attack/episode), die Achse ist hier also
        Notation, keine Semantik.

        Symptom-Grounds behalten Vorrang vor Disorder-Grounds, damit die Urteile
        der 37 Knoten mit Symptom-Ground bitgenau bleiben; Reihenfolge aus dem
        Graphen statt Set-Iteration (die war über PYTHONHASHSEED lauf-instabil,
        sobald ein Knoten mehr als einen Ground trägt)."""
        g = self.g
        props = g.props.get(nid, {})
        rel = props.get("rel")
        neg = bool(props.get("negated", False))
        carriers = temporal_carriers(g, nid)
        nmin, nmax = temporal_window(props)          # #1: Einheit inline nach Minuten
        if rel == "duration":
            s = duration_carrier(carriers, case)
            p = (case.duration(s, nmin, nmax,
                               props.get("bounds") or "[]") if s else K.UNKNOWN)
        else:
            p = case.relational(rel, set(carriers), nmin, nmax)
        return k_not(p) if neg else p

# =============================================================================
# Ergebnis-Aufbereitung (spiegelt die Cypher-Ergebnis-Query)
# =============================================================================
_STATUS = {K.TRUE: "erfüllt", K.FALSE: "offen", K.UNKNOWN: "unsicher"}
_BEWERTUNG = {K.TRUE: "RELEVANT", K.FALSE: "ausgeschlossen", K.UNKNOWN: "möglich"}

# --- Relevanz-Ranking (F2): ICHD-3-Fallback-Kategorien nachordnen ------------
# "Wahrscheinliche X" sind per Definition Auffang-Diagnosen (definite minus ein
# Kriterium) — nur zu kodieren, wenn die definite Diagnose NICHT erfüllt ist. Der
# Katalog-14-Auffang ("nicht anderweitig klassifiziert/spezifiziert") ist die
# letzte Instanz. Beide feuern als boolesch RELEVANT mit, sobald eine spezifische
# Diagnose passt, und verrauschen das Ergebnis. Sie werden NICHT umbewertet
# (bleiben strukturell RELEVANT), sondern nur nachrangig einsortiert.
# Anker am Wortanfang: trifft die "Wahrscheinliche/r"-Kategorien, aber NICHT
# Sekundärkopfschmerzen mit "... wahrscheinlich zurückzuführen auf ..." (z.B. 6.7.3.2).
_PROBABLE_RE = re.compile(r"^\s*(?:wahrscheinliche?[rs]?|probable)\b", re.I)
_CATCHALL_RE = re.compile(r"nicht anderweitig|nicht spezifiziert|nicht klassifiziert"
                          r"|not elsewhere classified|unspecified", re.I)


def _is_probable(name: Optional[str]) -> bool:
    return bool(name and _PROBABLE_RE.match(name))


def _is_catchall(code: str, name: Optional[str]) -> bool:
    return code.split(".")[0] == "14" and bool(name and _CATCHALL_RE.search(name))


def _rank_relevance(rows: List[dict]) -> None:
    """Setzt je RELEVANT-Zeile einen 'nachrang' (0 = vollwertig, 1 = Wahrschein-
    lich-Kategorie trotz definiter Diagnose, 2 = Auffangkategorie trotz spezifi-
    scher Diagnose) plus 'nachrang_grund'. Verändert die boolesche bewertung NICHT."""
    rel = [r for r in rows if r["bewertung"] == "RELEVANT"]
    has_definite = any(not _is_probable(r["name"]) and not _is_catchall(r["diagnose"], r["name"])
                       for r in rel)
    has_non_catchall = any(not _is_catchall(r["diagnose"], r["name"]) for r in rel)
    for r in rows:
        rang, grund = 0, None
        if r["bewertung"] == "RELEVANT":
            if _is_catchall(r["diagnose"], r["name"]) and has_non_catchall:
                rang, grund = 2, "Auffangkategorie — spezifischere Diagnose ist RELEVANT"
            elif _is_probable(r["name"]) and has_definite:
                rang, grund = 1, "Wahrscheinlich-Kategorie — definite Diagnose ist RELEVANT"
        r["nachrang"] = rang
        r["nachrang_grund"] = grund


def _is_letter_criterion(cid: str, did: str) -> bool:
    if not cid.startswith(did + "/"):
        return False
    return bool(re.fullmatch(r"[A-Z]", cid[len(did) + 1:]))

def _ancestor_diagnoses(g: RuleGraph, did: str) -> List[str]:
    out, seen = [], set()
    start = g.satisfied_by.get(did)
    stack = [start] if start else []
    visited: set = set()               # Zyklenschutz: Referenz-Kanten (nodes/missing)
    while stack:                        # koennen Diagnose->Diagnose-Zyklen bilden
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        for op in g.operands.get(nid, []):
            lbls = g.labels.get(op, set())
            if "Diagnosis" in lbls and op != did:
                if op not in seen:
                    seen.add(op); out.append(op)
                    if g.satisfied_by.get(op):
                        stack.append(g.satisfied_by[op])
            elif g.kind.get(op) is None:           # Criterion/SubCriterion
                if g.satisfied_by.get(op):
                    stack.append(g.satisfied_by[op])
    out.sort(key=lambda x: x.count("."))           # allgemein -> spezifisch
    return out

def classify(g: RuleGraph, memo: Dict[str, str], only_with_hits: bool = True) -> List[dict]:
    rows = []
    diagnoses = [nid for nid, lbls in g.labels.items() if "Diagnosis" in lbls]
    for did in sorted(diagnoses):
        dval = memo.get(did, K.UNKNOWN)
        kriterien = []
        # lokale Buchstabenkriterien
        letters = sorted([cid for cid, lbls in g.labels.items()
                          if ("Criterion" in lbls or "SubCriterion" in lbls)
                          and _is_letter_criterion(cid, did)],
                         key=lambda c: c.split("/")[-1])
        for cid in letters:
            logic = g.satisfied_by.get(cid)
            is_diff = logic is not None and g.kind.get(logic) == "DIFFERENTIAL"
            status = "unsicher" if is_diff else _STATUS[memo.get(cid, K.UNKNOWN)]
            kriterien.append({"buchstabe": cid.split("/")[-1], "kriterium": cid,
                              "art": "differential" if is_diff else "standard",
                              "status": status})
        # vererbte Eltern-Diagnosen
        for anc in _ancestor_diagnoses(g, did):
            kriterien.append({"buchstabe": "erbt " + anc, "kriterium": anc,
                              "art": "vererbt", "status": _STATUS[memo.get(anc, K.UNKNOWN)]})

        n_erf = sum(1 for x in kriterien if x["status"] == "erfüllt")
        n_off = sum(1 for x in kriterien if x["status"] == "offen")
        n_uns = sum(1 for x in kriterien if x["status"] == "unsicher")
        if only_with_hits and n_erf == 0:
            continue
        rows.append({"diagnose": did, "name": g.props.get(did, {}).get("name"),
                     "erfuellt": n_erf, "offen": n_off, "unsicher": n_uns,
                     "bewertung": _BEWERTUNG[dval], "kriterien": kriterien})
    # Relevanz-Ranking: Fallback-Kategorien (Wahrscheinlich/Auffang) nachordnen.
    _rank_relevance(rows)
    # Sortierung: RELEVANT zuerst, darin vollwertig vor nachrangig, dann Spezifität.
    order = {"RELEVANT": 0, "möglich": 1, "ausgeschlossen": 2}
    rows.sort(key=lambda r: (order.get(r["bewertung"], 3), r.get("nachrang", 0),
                             -r["diagnose"].count(".")))
    return rows

def print_rows(rows: List[dict]) -> None:
    for r in rows:
        tag = r['bewertung']
        if r.get("nachrang"):
            tag += "*"                       # nachrangig (Fallback-Kategorie)
        print(f"\n[{tag:14s}] {r['diagnose']:10s} {r['name'] or ''}")
        if r.get("nachrang_grund"):
            print(f"   ({r['nachrang_grund']})")
        print(f"   erfüllt={r['erfuellt']}  offen={r['offen']}  unsicher={r['unsicher']}")
        for c in r["kriterien"]:
            print(f"     - {c['buchstabe']:12s} {c['status']:9s} ({c['art']})  {c['kriterium']}")

# =============================================================================
# Beispielfälle
# =============================================================================
def example_cases() -> List[Case]:
    base_attrs = {("aura", "fully reversible"), ("aura", "visuell"),
                  ("aura", "unilateral"), ("aura", "positiv")}
    typical_absent = {("aura", "motor"), ("aura", "hirnstamm"), ("aura", "retinal")}

    # 1) Typische Aura MIT Kopfschmerz  -> 1.2 / 1.2.1 / 1.2.1.1 RELEVANT
    c1 = Case(name="typische Aura mit Kopfschmerz",
              present={"aura", "schmerz", "attacke"},
              attrs_present=set(base_attrs),
              attrs_absent=set(typical_absent),
              durations={"aura": (30, 30)}, counts={"attacke": 2},
              rels=[{"rel": "gradual_spread", "sym": "aura", "min": 5},
                    {"rel": "succession", "sym": "aura", "count": 2},
                    {"rel": "after", "from": "aura", "to": "schmerz", "within": 30}])

    # 2) Typische Aura OHNE Kopfschmerz -> 1.2.1.2 RELEVANT, 1.2.1.1 ausgeschlossen
    #    Kopfschmerz-Relation explizit als NICHT geltend markiert (FALSE, nicht UNKNOWN).
    c2 = Case(name="typische Aura ohne Kopfschmerz",
              present={"aura", "attacke"}, absent={"schmerz"},
              attrs_present=set(base_attrs),
              attrs_absent=set(typical_absent),
              durations={"aura": (30, 30)}, counts={"attacke": 2},
              rels=[{"rel": "gradual_spread", "sym": "aura", "min": 5},
                    {"rel": "succession", "sym": "aura", "count": 2}],
              rels_absent=[{"rel": "after", "from": "aura", "to": "schmerz"},
                           {"rel": "overlaps", "from": "aura", "to": "schmerz"}])

    # 3) Kopfschmerz-Bezug UNBEKANNT -> 1.2.1.1 und 1.2.1.2 'möglich' (Dreiwertigkeit!)
    c3 = Case(name="typische Aura, Kopfschmerzbezug unbekannt",
              present={"aura", "attacke"},
              attrs_present=set(base_attrs),
              attrs_absent=set(typical_absent),
              durations={"aura": (30, 30)}, counts={"attacke": 2},
              rels=[{"rel": "gradual_spread", "sym": "aura", "min": 5},
                    {"rel": "succession", "sym": "aura", "count": 2}])
              # keine rels_absent -> after/overlaps bleiben UNKNOWN

    # 4) Aura MIT motorischer Komponente -> 1.2 RELEVANT, 1.2.1 ausgeschlossen
    c4 = Case(name="Aura mit motorischer Komponente (nicht typisch)",
              present={"aura", "schmerz", "attacke"},
              attrs_present=set(base_attrs) | {("aura", "motor")},
              durations={"aura": (30, 30)}, counts={"attacke": 2},
              rels=[{"rel": "gradual_spread", "sym": "aura", "min": 5},
                    {"rel": "succession", "sym": "aura", "count": 2},
                    {"rel": "after", "from": "aura", "to": "schmerz", "within": 30}])
    return [c1, c2, c3, c4]

# =============================================================================
# Selbsttest-Fixture (OHNE Neo4j) — prüft Engine, Negation, Dreiwertigkeit
# =============================================================================
def selftest_graph() -> RuleGraph:
    """Mini-DAG, der das Negations-/Dreiwertigkeits-Verhalten von 1.2.1.2
    nachstellt: TEST  AND( B/1=GROUND(aura,visuell) , B/2=NOT-after(aura->schmerz) )."""
    return RuleGraph.from_dict({
        "kind": {
            "TEST": None, "TEST/A": None, "TEST/B": None,
            "TEST/B#1": None, "TEST/B#2": None,
            "n0": "AND", "nA": "GROUND", "nB": "AND",
            "n1": "GROUND", "n2": "TEMPORAL",
        },
        "labels": {
            "TEST": {"Diagnosis"}, "TEST/A": {"Criterion"}, "TEST/B": {"Criterion"},
            "TEST/B#1": {"SubCriterion"}, "TEST/B#2": {"SubCriterion"},
        },
        "props": {
            "n2": {"rel": "after", "min": 0, "max": 60, "negated": True},
        },
        "operands": {"n0": ["TEST/A", "TEST/B"], "nB": ["TEST/B#1", "TEST/B#2"]},
        "grounds": {
            "nA": [("aura", "Symptom", False), ("visuell", "Attribute", False)],
            "n1": [("aura", "Symptom", False), ("visuell", "Attribute", False)],
            "n2": [("aura", "Symptom", False), ("schmerz", "Symptom", False)],
        },
        "satisfied_by": {
            "TEST": "n0", "TEST/A": "nA", "TEST/B": "nB",
            "TEST/B#1": "n1", "TEST/B#2": "n2",
        },
    })

def run_selftest() -> int:
    g = selftest_graph()
    r = Reasoner(g)
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {got} (erwartet {want})")

    # after-Fakt vorhanden -> ¬after = FALSE -> B FALSE -> TEST ausgeschlossen
    c = Case(present={"aura", "schmerz"}, attrs_present={("aura", "visuell")},
             rels=[{"rel": "after", "from": "aura", "to": "schmerz", "within": 30}])
    check("mit Kopfschmerz -> TEST", r.evaluate_case(c)["TEST"], K.FALSE)

    # after explizit abwesend -> ¬after = TRUE -> B TRUE -> TEST RELEVANT
    c = Case(present={"aura"}, attrs_present={("aura", "visuell")},
             rels_absent=[{"rel": "after", "from": "aura", "to": "schmerz"}])
    check("ohne Kopfschmerz -> TEST", r.evaluate_case(c)["TEST"], K.TRUE)

    # after unbekannt -> ¬after = UNKNOWN -> B UNKNOWN -> TEST möglich
    c = Case(present={"aura"}, attrs_present={("aura", "visuell")})
    check("Kopfschmerz unbekannt -> TEST", r.evaluate_case(c)["TEST"], K.UNKNOWN)

    print("\nSelbsttest:", "alle bestanden" if fails == 0 else f"{fails} fehlgeschlagen")
    return 1 if fails else 0

# =============================================================================
# main
# =============================================================================
def main() -> int:
    if "--selftest" in sys.argv:
        return run_selftest()

    from ichd3.engine.db import Neo4jConfig, get_driver

    cfg = Neo4jConfig.from_env()
    try:
        driver = get_driver(cfg)
    except RuntimeError as e:
        print(e)
        return 2
    try:
        g = RuleGraph.from_neo4j(driver, cfg.database)
    finally:
        driver.close()

    print(f"Regelgraph geladen: {sum(1 for k in g.kind.values() if k is None)} Entitäten, "
          f"{sum(1 for k in g.kind.values() if k is not None)} Logikknoten.")
    reasoner = Reasoner(g)
    for case in example_cases():
        print("\n" + "=" * 78 + f"\nFALL: {case.name}\n" + "=" * 78)
        memo = reasoner.evaluate_case(case)
        print_rows(classify(g, memo, only_with_hits=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
