# Gain Guard V2 strict: failed to improve the fixed baseline

User-run review completed; local artifact audit PASS. The primary regional_guarded policy fails the expansion gate. No additional training/inference/Kaggle notebook was launched during this review. Results below are on the same512 development validation images, not the1000-image SCST subset.

| Policy | CIDEr | CHAIRs | Hallucinated captions / mentions | Changed versus fixed |
|---|---:|---:|---:|---:|
| Fixed baseline | 1.263698 | 3.1250% | 16 / 17 | 0 |
| Scores raw / guarded | 1.263698 | 3.1250% | 16 / 17 | 0 |
| Global raw | 1.251245 | 5.2734% | 27 / 29 | 254 |
| Global guarded | 1.262630 | 3.1250% | 16 / 17 | 1 |
| Regional raw | 1.241589 | 5.4688% | 28 / 30 | 256 |
| Regional guarded (primary) | 1.262488 | 3.3203% | 17 / 18 | 2 |

## What happened

The gain predictor does not generalize sufficiently to this development subset. Regional raw changes256 captions:122 improve per-imageCIDEr,127 decline and7 tie, with netCIDEr−.022109. It removes2 hallucinated captions but introduces14. Global raw improves125, declines122, ties7;netCIDEr−.012453 and2 hallucination fixes versus13 introductions. Positive change counts alone therefore do not imply a gain; magnitudes and error types matter.

Guard calibration is conservative:global correction.794141 andregional.699616 block most proposals, leaving only1 and2 changes. However, every retained change decreasesCIDEr, and regional introduces1 hallucinated caption. Thus this is not merely a case of a useful ranker being prevented from improving by a high threshold. Lowering the threshold without new evidence would expose the already harmful raw policy.

Scores-only selects epoch0, reproducingfixed. Global/regional both selectepoch6 according to the locked tune128 rule. Training fit1024,epoch-tune128,calibration128 andvalidation512 IDs are disjoint;validation refs were not supplied as modelinputs. Calibration nominal90% error coverage is not a90% guarantee of good switches, consistent with the method's stated limitation.

## Retained switches

Bothglobal/regional changeimage5072:

- Fixed: `a clock hanging from a wall next to a window.`
- Chosen: `a clock hanging on a wall next to a bottle of wine.`
- Predictedgain .929935global/.806518regional, exceeding the respective corrections. Actual per-imageCIDEr change−.546989 under validation-IDF.

Regional also changesimage33157:

- Fixed: `a cat sitting on the floor in front of a window.`
- Chosen: `a cat standing in a living room next to a dog.`
- Predictedgain .731143; actual per-imageCIDEr change−.072591 under validation-IDF.

Train-IDF predicted gains and validation-IDF evaluation gains use different corpora; numerical differences here are illustrative, not directly calibrated residual estimates. No image-level physical-truth adjudication was performed locally.

## Verification and limits

Localaudit `audit_caption_gain_guard.py` checks archiveCRC,protocol/policy/fit-IDF hashes,cohort identities,strict-loads all3smallheadweights usingweights_only=True,finiteparameters,epoch selection,baseline gain0,proposal/guard arithmetic,traincalibrationquantile,policy caption membership,officialPTB/CIDEr/BLEU/ROUGE for7policies andCHAIR reaggregation. METEOR cross-checked only. Allchecked artifacts consistent;the auditPASS is not a performancePASS.

Featurecache was regenerated. Frozenfeaturetensors are absent fromreview,so model forward/featurehashes cannot be independentlyreplayed. Fulltrainreferences absent,so lockedIDF/target generation not independentlyrebuilt;storedcalibration arithmetic verified. Reviewcontainsheadweights,notQFormerweights. Runtime reportsTeslaT4/PyTorch2.11.0+cu128 withstrictdeterministicalgorithms andmathattention. Reusedvalidation512 remainsdevelopmentdata,notindependentconfirmation.

Source `caption_gain_guard_v2_strict_review.zip`,3830113bytes,SHA256 `d13af9ca31ea70e5b08a1aafc9069c6d6bc6ce861f52a8af23ba4d17ab870bc6`.

## Decision

Do not promote this selector, lower its guard based on these512 answers, increaseepochs or expandtraining as if it had passed. Keep the fixed selector and the separate SCSTcount0.5 checkpoint as references. Oracleheadroom was real, but learning a useful selector from this smallweaklysupervisedpilot did not succeed;headroom alone was insufficient evidence.

The next architecture/training proposal must establish that its alignment signal can distinguish actual candidate substitutions, including addedobjects and changedactions, before another expensive rankingrun. Merely repackaging the sameCLIPfeatures with a differentthreshold does not address the observed failure. No new experiment notebook is issued by this result review.
