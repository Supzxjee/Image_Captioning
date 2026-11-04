# Historical QFormer versus current CSA+SCST2500

04/11/2026. User requests practical comparison. [Numeric record](QFORMER_VS_CSA_CURRENT.json). No new model/inference/official scorer run. This compares delivered systems with unequal training/input/decoder settings, not an isolated QFormer-versus-CSA architecture experiment.

## VALID5000 object metrics from existing reports

| Metric | Original QFormer raw | QFormer+SCST/CE/object raw | Current CSA+SCST/CE/object2500 |
|---|---:|---:|---:|
| CHAIRs ↓ |4.28%|**3.88%**|5.02%|
| Hallucinated captions ↓ |214|**194**|251|
| CHAIRi ↓ |2.9793%|**2.6780%**|3.7080%|
| False object mentions ↓ |216|**196**|258|
| Object recall ↑ |42.9119%|**43.5271%**|39.8104%|
| Recovered class/image pairs ↑ |6835|**6933**|6341|
| GT class/image denominator |15928|15928|15928|

QFormer totals recombined from the audited disjoint tune1000 and validation-remainder4000 of [historical full-train report](../../SCST_FULL_TRAIN_W05_RESULT.md), using originalrank0 or selectedobjectEND rank0 consistently. Checked the six input rates against counts per policy, then summed numerators/denominators (not averages of percentages). Full5000 has known KarpathyVALID provenance from prior audit; original per-image QFormer IDs/prediction payloads are not re-opened here, so this is an aggregate historical comparison, not new paired image-level confirmation. QFormer END was selected on tune1000; the2500CSA endpoint was fixed. Do not call entireVALID an untouched independent evaluation.

The older QFormer object system has57fewer flagged captions/62fewer falsementions and592more correct class/image pairs at the recorded fullVALID operating points. This is a substantive quality target for currentCSA: reduce errors while recovering coverage, not simply omit nouns. CurrentCSA improved against its own XE15 base but has NOT surpassed this stronger historical reference on these object metrics.

## Caption scores: not a matched CIDEr subtraction

- QFormer+object **raw**, VALIDremainder4000: CIDEr119.5705 on×100scale, BLEU4.374134, ROUGE-L.573309.
- CurrentCSA+object, fullVALID5000: CIDEr103.2307, BLEU4.341330, ROUGE-L.540322.
- Different cohort/reference sets and CIDEr IDF: do NOT subtract these to claim a16.34point architecture deficit. QFormer's fixed-reranked123.8974 is a further different policy, not a fair rawcomparison.
- CurrentCSA has no TEST result at2500; do NOT compare itsVALID score with old QFormerTEST121.2024raw/124.7342reranked as a matched test ranking.
- The prior saved-caption [common4000 comparison](SAVED_QUALITY_RESULT.md) showed historical QFormerA/C0/C ahead of CSA9/10, but these are different checkpoints and greedy-vs-beam policies; not a current2500 result.

## Training and loss interpretation

QFormer object continuation saw113287uniqueTRAIN images/14161SCSTupdates after its own trained baseline. CurrentCSA2500 sees10000uniqueTRAIN images/20000events/2500updates after15XEepochs. CurrentCSA's largerXEepoch number does not equal matched feature/decoder/pretraining/RL exposure. Do not conclude CSA intrinsically inferior, or guarantee that fullTRAIN closes the gap. Historical vocab/tokenizer/input/decoding differ; total losses or rawNLL across these systems are not comparable. WithinCSA, VALIDNLL2.405520→2.374963 is meaningful.

## Recommendation

Keep QFormer+object as a strong practical reference and currentCSA as the candidate still to improve. Before adding QFormer to CSA or attributing the gap to a particular module, compare saved captions on one commoncohort with identical references/scoring, and label any decoding-policy difference. Saved-caption CPU scoring can avoid retraining. No new scoring notebook, GPU/training or architecture change is authorized by this comparison. JointCIDEr/CHAIR/recall, not novelty, remains the user goal.
