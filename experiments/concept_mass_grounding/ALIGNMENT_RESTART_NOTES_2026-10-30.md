# Restart alignment brainstorm — user goal retained

30/10/2026. User explicitly rejects replacing the original research goal with an empirical-comparison paper: “bây giờ phải có một phương pháp hay đóng góp gì đó tới căn chỉnh cơ bạn ơi”. Authorization now is to brainstorm a substantive visual–semantic alignment contribution, then discuss/record a choice before implementation. No new training/data extraction/GPU budget authorized. Closing C/G/C0 and not tuning on extra4000 remains in force; it does not close the overall research goal.

## Required proposal

State one concrete correspondence being learned, the module and exact inputs/outputs, supervision available without test leakage, how the actual caption decoder uses the correspondence at inference, the smallest falsifiable pilot and a fair baseline. Include language-only/geometry shortcuts where relevant and why earlier failures do not already refute the hypothesis. A useful task-specific adaptation can be a contribution; do not require a globally new mathematical primitive. Also do not claim novelty from a new name, weak control, implementation work alone or a generic combination of familiar losses.

## Verified local lessons

- C/G/C0: quality gains retained without proposed preservation/suppression; no established added factuality. Preserve evidence, stop the current claim and no more tuning on locked4000.
- I/J: J beat its matched degrading control but remained below A; panel did not establish semantic compensation. Reader-off was not a full bypass. Does not prove all learned alignment impossible.
- Attention-geometry138TRAIN diagnostic: noun-specific spatial signal beyond image mean/derangement; centroid hits only28.26%. GT-informed max-Q oracle suggests selection capacity but does not prove attainability or caption utility.
- Frozen A/T/W correction138: increasing correct-box attention mass did not improve teacher-forced target-noun utility; no-go for that correction. Attention mass alone is not sufficient evidence of useful alignment.
- Prior interchange/shared-pair proposals had close known primitives (RAVEL/MDAS and segment/role-aligned generation). This is not proof every task-specific extension is identical, but novelty delta must be concrete.

## Primary literature checked in this round

Academic MCP search tools were not available. Used primary paper search/retrieval via web; CVF PDF requests returned403, so author arXiv versions were used for methods. This is a focused nearest-method check, not an exhaustive novelty search or quartile audit.

1. Zhou et al., **More Grounded Image Captioning by Distilling Image-Text Matching Model**, CVPR2020.
   - Primary PDF: https://arxiv.org/pdf/2004.00390
   - CVF metadata: https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_More_Grounded_Image_Captioning_by_Distilling_Image-Text_Matching_Model_CVPR_2020_paper.html
   - Read §3.3 Eq15–19 and associated method text, not claiming a complete new reread of all experiments. Frozen POS-SCAN guides captioner noun-region attention via KL; GT attention supervision variant viaNLL; matching score used withCIDEr in SCST. Teacher can be removed at inference. Thus “distill matching attention into captioner” is inherited, not itself new.
2. Xiao et al., **FLAIR: VLM with Fine-grained Language-informed Image Representations**.
   - Primary full HTML: https://arxiv.org/html/2412.03561v1
   - CVPR2025 paper path: https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_FLAIR_VLM_with_Fine-grained_Language-informed_Image_Representations_CVPR_2025_paper.pdf (fetch403 in this round).
   - Read §3.2–3.3 negative-pair construction and Eq2. Text-conditioned image pooling is established. A negative pairing that can be solved by comparing the text condition with text embedding is a shortcut; holding the text condition while contrasting mismatched images is relevant. Retrieval/local representations do not automatically train caption use; a generated future phrase is unavailable at decoding time.
3. **Align Where the Words Look: Cross-Attention-Guided Patch Alignment with Contrastive and Transport Regularization for Bengali Captioning**, arXiv2509.18369v1,22Sep2025.
   - Primary full HTML: https://arxiv.org/html/2509.18369v1
   - Read §4.6 Eq8–11 and nearby OT/method text. Decoder-attention-weighted real/synthetic descriptors, cosine PAL, InfoNCE and Sinkhorn OT. Venue/peer-review status unverified; do not present as an established Q2+ journal paper. It is a relevant preprint against a generic “attention+contrast+OT” originality claim, not evidence that every new alignment design is already solved.

Astra was asked for one concrete alignment-focused candidate, a decisive small pilot and honest novelty boundaries; no ZIP rereads, neural runs or code requested. User has not accepted the empirical-paper scope. Decision pending; update RESEARCH_STATE after response.

## Additional checks while Astra drafts the candidate

- Read `coco_grounding_feasibility/routing_model.py`: R already supervises the native decoder→Q→patch path with noun/box mass loss at .05 alongside CE/SCST. Simply proposing that composed path again is not a new project contribution. BR_RESULT retains the inconclusive R−B/R−A pilot evidence.
- Read `noun_evidence_proposal/README.md` and `GS_RESULT.md`: old G/S train a patch/class head (S with bbox pooling supervision), but both globally pool at inference and add a bounded noun-logit bias. S has worse CIDEr and more hallucinations than A. A prefix-conditioned region pointer at inference would differ from that implementation; its benefit remains untested.
- Neural Baby Talk (CVPR2018), https://arxiv.org/pdf/1803.09845: read §3.1–3.2 and Eqs3–10. Region pointing, a visual sentinel and region-linked lexical generation are established. A pointer or NULL gate by itself cannot be claimed new.
- Astra's interim direction is a region–word pointer at output, with same-prefix object probability mass preserved while redistributing among object words. This is an **unapproved candidate**, not a finalized protocol or successful method. Required substantive distinction: prefix-specific visual correspondence used in the actual lexical distribution, rather than only a global class bias or an attention visualization.
- Root critique sent to Astra: preserving object mass at a fixed prefix does not preserve total noun mentions or caption length under autoregressive rollout; detector labels can become a class-prior shortcut; compare local ROI evidence against a matched global/class-information control. Training boxes must not be required as oracle input on unseen images. Multi-instance noun references require a positive set, not a fabricated unique instance label.

## Source availability handoff after Argon brief

Astra confirmed that a shared appearance term in a linear/dot-product key cancels in relative ROI scores, although ROI-vs-NULL mass may still depend on it. The original control needs revision. Astra also treats hard lexical-mass preservation as an unvalidated safeguard; a two-arm comparison with the same constraint cannot establish its added benefit. A three-arm design is being considered, not authorized or implemented.

Root checked `C:/KhoaLuanTotNghiep/Paper` and repository filenames: no local Neural Baby Talk/copying/CBS PDF was found in those locations. This is not an exhaustive search of the user's disk. A fresh official/arXiv primary-page tool request returned503 as well; do not count this failed retrieval as reading new papers.

NBT method handoff (source read earlier via primary web PDF, **paraphrase notes, not a saved full text**): §3.1 defines a content-based region pointer from recurrent state and ROI features (Eqs4–5). It adds a visual sentinel for the text branch (Eqs6–9); Eq10 is the text vocabulary softmax. §3.2 lexicalizes pointed visual slots using detector categories, plurality and fine-grained naming. Thus prefix/state→ROI pointing, a fallback branch, and explicit region-linked word generation are inherited mechanisms. These verified notes do not establish whether the proposed exact mass constraint is absent from NBT or every later copy model. Primary URL: https://arxiv.org/pdf/1803.09845 .

No nearest-full-text comparison for DCC/Guided Open Vocabulary/CBS was completed in this follow-up. Required prior-reading gaps remain explicit; do not turn network failure into a novelty claim or run approval.

**Later same-day update:** Astra reports successful primary NBT PDF retrieval and work on NBT/DCC/CBS/M3ID under `tmp/pdfs/nearest_pointer_20261030`, using Python network access. Retrieval is not itself method reading; await Astra's section/equation-backed comparison rather than repeating the downloads. User PDFs are not needed at this point. The earlier503 report remains a record of the earlier tool attempt, not a continuing claim that sources are inaccessible.

A stronger mathematical limitation was identified during the exchange: same-prefix fixed object mass does not even prevent a new greedy EOS decision. With three single-token classes, base probabilities(.4,.1,.1), EOS=.3, other=.1, redistribute the same noun mass.6 to(.2,.2,.2). EOS becomes greedy without increasing its probability. This refutes an unconditional no-shortening/no-EOS-evasion motivation, not the stated algebraic probability invariants. Root sent this example to Astra and added it to the Argon brief. No new model experiment was performed.
