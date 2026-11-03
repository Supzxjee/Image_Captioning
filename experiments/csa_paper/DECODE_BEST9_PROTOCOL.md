# Frozen CSA epoch9: greedy versus saved beam5

Approved by user on03/11/2026 after saved-quality review. Question: does changing search width alone improve the current checkpoint's delivered captions? No training, length normalization, minimum length or parameter sweep.

Use exact Output139 best.bin SHA f09e11c374477be93cc142a238dbd0e206060f33fa4fab5988e767ea7d259fe6, unchanged model source, FP32/eval/TF32off. Use only fullROI134 validation shards (5000images), not TRAIN staging. Frozen bank buffers are already in the checkpoint; a separate bank Dataset is unnecessary. Exact original review contains beam5 captions, references, vocabulary and input identity. Validate file hashes and checkpoint identity before inference.

Generate greedy with existing beam_search(width1), maximum20 including EOS, same PAD/BOS bans and raw scores. Encode batches8, original rank-strided order, exactly as training validation. Two independent workers use two GPUs; no gradient/DDP collectives. Replay beam5 on the first original batch8 per rank as input/model/decoder guard, comparing content token IDs and score (rtol1e-5/atol1e-4). Do not regenerate full beam5. Sample agreement supports replay fidelity, not exhaustive bitwise equivalence across hardware. Preserve failure diagnostics; no partial-cohort headline.

Full5000 and common4000 metrics use same evaluator inputs/order as completed saved-quality V2. Reuse existing common_refs, chair_counts and CIDEr score alignment. Recompute beam baseline too; replay all headline metrics against archived values within1e-7 and raw counts exactly before declaring COMPLETE. CIDEr/BLEU4/ROUGE-L, CHAIRs/i, recall, raw counts, length/empty/≤2word counts and changed captions. No old-model reranking, no new test. This is a reused development diagnostic, not a novel alignment mechanism.

Notebook cap90minutes total worker phase (not an ETA or budget guarantee). Stage metadata/NPZ only; no image/model downloads, no optimizer. User runs Kaggle; local checks only AST/text/NumPy fixtures. Keep best9 and latest10 untouched. Future15epoch XE/module/RL requires separate execution preparation/agreement; this notebook does not launch any training.
