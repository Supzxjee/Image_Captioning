# CSA saved-caption quality — COMPLETE, 03/11/2026

CPU V2 review: `csa_saved_quality_v139_v2_review.zip`, SHA256 `274f0277f29ca201d3b945889faa79daf765f385925c3f3c607389bdf93517cc`. Runtime source `3a6532941a98384a8501e7dcfb5be3453ec407ab`. All historical metric replays passed; CIDEr and CHAIR comparison flags are valid. No checkpoint load or new generation. Seven small runtime receipts are in `saved_quality_review_text/`; audited numeric output is [SAVED_QUALITY_RESULT.json](SAVED_QUALITY_RESULT.json).

## Common 4,000 development images

Same image IDs and exact historical first-five references; percentages below. A/C0/C use their saved greedy captions; CSA uses saved beam5 captions.

| Model | CIDEr | BLEU4 | ROUGE-L | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Hallucinated captions | False mentions | Mean words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1.189771 | .352651 | .575557 | 5.975% | 3.917% | 46.809% | 239 | 254 | 10.129 |
| C0 | 1.204560 | .358872 | .578636 | 5.875% | 3.814% | 47.108% | 235 | 248 | 9.995 |
| C | 1.207042 | .360628 | .579111 | 5.750% | 3.752% | 47.030% | 230 | 243 | 9.938 |
| CSA epoch 9 | 1.017741 | .338524 | .537870 | 5.800% | 4.160% | 40.316% | 232 | 246 | 8.262 |
| CSA epoch 10 | 1.007737 | .333143 | .534579 | 6.275% | 4.474% | 40.450% | 251 | 260 | 8.087 |

CSA9 − C CIDEr = −0.189301; paired 10,000-bootstrap 95% CI [−0.211529, −0.167228]. Of C's hallucinated captions, CSA9 repairs166 but introduces168 elsewhere; exact McNemar p=.956374. This does not establish hallucination improvement or equivalence. Recall decreases6.714 percentage points:5,110 versus5,961 recovered class/image pairs out of12,675. These counts are class/image coverage, not individual object instances. Shorter captions and lower recall indicate omission; its causal source is not established.

Other paired CIDEr CIs: CSA9−A = −.172030 [−.194132, −.149615]; CSA10−CSA9 = −.010004 [−.026376, .006869]. Bootstrap describes this reused development set, not uncertainty across training seeds.

## Full 5,000 validation images

These use25,010 references and a different cohort/IDF, so their CIDEr must not be mixed with the common4000 table.

| Model | CIDEr | CHAIRs | CHAIRi | Recall | Hallucinated captions | False mentions |
|---|---:|---:|---:|---:|---:|---:|
| CSA9 | 1.002933 | 5.660% | 4.095% | 40.149% | 283 | 303 |
| CSA10 | .997179 | 6.160% | 4.388% | 40.225% | 308 | 318 |

Keep epoch9 `best.bin`, selected by validation CIDEr before this CHAIR comparison. Epoch10 remains the latest resumable checkpoint, not the preferred quality checkpoint.

## Loss and scope of the conclusion

Epoch9→10: observed TRAIN NLL2.441058→2.417765; full validation NLL2.446409→2.435452. Loss decreases while CIDEr point estimate decreases and hallucinated-caption count increases. This alone is not proof of overfitting. Comparable historical validation NLL is absent; old SCST/KL objectives and vocabularies differ (only6,916 shared word types despite10,000 entries each). Do not compare their raw total losses.

Current CSA is a fresh10-epoch XE adaptation; historical C is a pretrained/SCST continuation, with different decoding. The results compare current delivered pipelines, not an isolated architecture effect or final CSA potential. CHAIR measures object hallucination, not all relation/factual errors; no new test evaluation occurred.

Recommendation: first propose a small greedy-versus-existing-beam5 diagnostic on the same best9 checkpoint, given409/5000 captions of at most2 words. Decoding is only a hypothesis. Further15-epoch XE/RL+CE work and the image-conditioned bank candidate require user agreement; no new GPU experiment is authorized by this review.

Verification: metadata-only audit passed78 grouped checks and five Git source hashes; recomputed counts, lengths, CIDEr means, paired bootstrap and exact McNemar from stored results. No local Torch/model/scorer execution. Original review ZIP and large parsed cache remain local/ignored.
