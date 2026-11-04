# CSA word–region supervision: approved data feasibility step

2026-11-04 UTC+7. User: “thôi thực hiện đi”. Implement the CPU data audit first,
as required by X070; do not spend GPU before inspecting labels. No trainer is
authorized to run automatically. This step does not change the existing model.

Prior R (`../coco_grounding_feasibility/BR_RESULT.md`) already trained a noun →
decoder → Q → patch box-mass objective. R−continuation CIDEr +.006550 with CI
[-.015988,.029683] did not establish benefit. This is a reopening of that family
on CSA's direct object-region attention, not an untried idea or new loss.

Freeze 10,000 Karpathy TRAIN/restval IDs by SHA256 of
`csa-grounded-attention-v1/42/<filename>`. Reuse original v134 ROI boxes/offsets;
never reconstruct detector order or encode images. Verify the existing manifest,
vocabulary and every accessed shard against v134 transport hashes. Read only box,
ID and offset arrays, not feature arrays. Official COCO train/val2014 annotations
are streamed and filtered to these TRAIN IDs (restval can originate in val2014).
Both annotation members have previously verified hashes. No TEST/VALID labels
retained. No checkpoint, bank, Torch, GPU, optimizer or scorer needed.

Use a conservative lexicon: exact COCO names and explicit plural forms; a small
declared person alias list. Longest compound first; ungroundable compounds such as
“hot dog” block a shorter “dog” match. One final-token target per mention; verify
next-token index against the original 54-token sequence and skip UNK/truncation.
This is weak category-level matching, not exact phrase-instance truth. Report
coverage of mapped mentions and fractions of all tokens/captions separately;
unmatched tokens are not counted as proven negative or as recognized nouns.

Positive proposals overlap ANY noncrowd GT box of that class at IoU >= .5.
Skip a mention if category has crowd, missing GT, no positive, fallback, UNK or
truncation. Store ALL positives, never choose an instance arbitrarily. Metadata
and image dimensions must agree exactly; box corruption fails, not silent skip.

Outputs: sealed cohort/lexicon/annotation provenance, full mention JSONL labels,
per-class coverage/skip counts, and a deterministic 100-case review HTML/JSON.
Review includes eligible and rejected cases; a second 100 eligible-case sample
supports the proposed 95/100 positive-set check. Draw cached proposal boxes and
GT on externally loaded COCO photos; no images are fetched locally by this agent.
Missing browser image is “unreviewable”, never confirmed. Human review stays
unfilled. No automatic GO based on coverage or assistant-generated labels.

Proposed eligibility gate: >=80% of lexicon-mapped, noncrowd GT-supported mentions
have positive proposals, plus >=95/100 manually verified eligible examples. Report
all denominators. A complete audit means preparation complete, not training-ready.
After review, prepare matched XE vs XE+.1 grounding, 2,000 updates per arm from
CSA10, fresh Adam LR1e-5, global32, same order and attention implementation. These
pilot settings remain provisional until label review and current GPU budget.
