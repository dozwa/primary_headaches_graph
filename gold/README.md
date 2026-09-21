# Gold-2 cases for the primary graph (chapters 1–4)

Generated on 18 Sep 2026 from the graph in `../graph/` (reasoner fingerprint `3e4abc821591`) by the
Gold-2 generator of the source pipeline: for every diagnosis, positive cases are constructed from
the criteria (12 structural variants × 80 seeds, de-duplicated), near-miss cases drop one feature of
a positive case, refuted cases falsify one criterion. Every case is evaluated against all 77
diagnoses in `scripts/gold2_crosstalk.py`; the target verdicts are the poster's rule test.

| tier | n | construction | expected verdict |
|---|---:|---|---|
| `pos_native/` | 788 | all criteria of the target satisfied, absences stated explicitly | TRUE |
| `contra/` | 298 | one criterion falsified (`falsified_criterion`), 6 per diagnosis where possible | FALSE |
| `nearmiss/` | 1 530 | one feature of a positive case removed (`removed_feature`, `source_case`), 2 per case | not TRUE (UNKNOWN under open world) |

`fig3_case.json` is the example case of the poster's Fig. 3 (ten attacks in the past year, six to
ten hours, unilateral, pulsating, photo- and phonophobia; nausea, intensity and behaviour on exertion
not stated).

All cases are synthetic; the `evidence` strings are construction notes, not text from any patient.
The data follow the ICHD-3 terms of the International Headache Society (`../LICENSE-DATA`).
