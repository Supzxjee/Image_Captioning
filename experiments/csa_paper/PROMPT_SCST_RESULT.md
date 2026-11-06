# Prompt CSA + SCST/CE + fixed reranking — VALID5000

06/11/2026. CONTROL SaveVersion180, PROMPT SaveVersion60 (user confirmed). Both completed one113287-image TRAIN epoch /14161updates; original/END fullVALID5000, noTEST.

| Arm / phase / selection | CIDEr x100 | BLEU4 x100 | ROUGE-L x100 | CHAIRs % | CHAIRi % | Recall % | Flagged captions | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL/original/decoder_rank0 | 99.4587 | 32.9878 | 53.2696 | 6.6000 | 4.8236 | 39.7664 | 330 | 2.448211 |
| CONTROL/original/fixed_rerank | 107.5807 | 34.6053 | 55.3978 | 5.2000 | 3.4185 | 42.8428 | 260 | 2.448211 |
| CONTROL/end/decoder_rank0 | 105.1821 | 34.7785 | 54.2192 | 5.5000 | 4.0437 | 40.3001 | 275 | 2.397422 |
| CONTROL/end/fixed_rerank | 114.2522 | 36.4487 | 56.5535 | 4.4600 | 2.9824 | 43.5962 | 223 | 2.397422 |
| PROMPT/original/decoder_rank0 | 98.8453 | 32.3083 | 52.9931 | 4.4000 | 3.1451 | 40.5387 | 220 | 2.516285 |
| PROMPT/original/fixed_rerank | 107.8530 | 34.4522 | 55.5554 | 3.2400 | 2.0724 | 44.3056 | 162 | 2.516285 |
| PROMPT/end/decoder_rank0 | 105.8365 | 34.8481 | 54.3473 | 4.0600 | 2.8552 | 41.4427 | 203 | 2.423685 |
| PROMPT/end/fixed_rerank | 113.8004 | 36.2831 | 56.5679 | 3.0200 | 1.9274 | 44.5944 | 151 | 2.423685 |

PROMPT alone: CIDEr98.8453 ->105.8365 after SCST+CE, then113.8004 after reranking. Flagged captions220->203->151; total CIDEr+14.9551points,69fewerflags (-31.36%). No extra object reward was used.

At END before reranking, PROMPT vs CONTROL: CIDEr+0.6545points, CHAIRs5.50->4.06%(275->203), recall40.3001->41.4427%. After fixed rerank: CIDEr114.2522->113.8004 (-0.4518points), CHAIRs4.46->3.02%(223->151;72fewer/-32.29%), CHAIRi2.9824->1.9274%, recall43.5962->44.5944%. Thus prompt retains an object-hallucination benefit; no supported CIDEr superiority.

Conditional2000paired-bootstrap95%CI for PROMPT-CONTROL CIDErpoints: rank0[-0.7352,2.0864],rerank[-1.7919,0.9649]; both include0, not equivalence proof. CHAIRsdelta=-1.44pp for both, CIs rank0[-2.08,-.82],rerank[-2.02,-.90]. One seed, fixed corpus IDF and reusedVALID; not training variability.

Old CSA15+SCST+CE+rerank VALID115.2416/CHAIRs4.12% remains a higher CIDEr reference; newPROMPT113.8004/3.02% has lower object hallucination. Different initial XE epochs and experiment histories prevent isolated prompt-effect attribution.

Recommendation: preserve PROMPT60 and CONTROL180 full outputs, checkpoints and candidates. Freeze current recipe before any user-approved TEST; do not automatically train more, change selector weights or choose a checkpoint posthoc. Prompt text vs adapter/ROI/capacity contribution and relation correctness remain unverified.

Audit: source Git pins, complete cursors, sealed evaluation hashes, original metric replay, caption coverage, CHAIR counts/input, per-image CIDEr means and fixed-choice candidate membership. Full selector-choice replay is recorded in JSON audit. No author models/Torch/official scorers/inference; binary weights and rawJPEGs are not independently available in review ZIP. NLL is the same model teacher-forced loss before/after reranking, not likelihood of selected captions. See PROMPT_SCST_RESULT.json and audit_prompt_scst_result.py.

Audit completed:162checks PASS, including replay of all20000 fixed selector choices from saved CLIP scores/detections; no model/scorer rerun. CONTROL180 andPROMPT60 source versions confirmed.
