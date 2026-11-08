# Paper analysis — 08/11/2026

User approved analyses of existing outputs before new training. Freeze CONTROL180/PROMPT60 weights, TEST captions184/61, metric outputs185/62. No selection/model/selector changes based on TEST. RegionCLIP and object-only training deferred pending these analyses.

Deliver: (1) own-account CPU notebooks replay SPICE All and report Objects/Relations/Attributes on unchanged normal/fixed_rerank captions; exact SHA-verified input manifest inherited from PROMPT_TEST_EXTRA_INPUTS.json. Undefined category scores become null, never zero; report finite denominators and paired-common-denominator differences. (2) existing parameter/training/decode/rerank costs with excluded setup/extraction explicit. (3) deterministic CHAIR-based caption shortlist, including repaired/introduced/unchanged flags; automated flags are not visual truth and require human review.

No model, official metric, notebook or GPU run by author. Static/mock/data checks only. User runs CPU notebooks. SPICE categories are caption scene-graph agreement, not pixel grounding accuracy. Prior TEST exposure, one seed, C10/P9 start and extra adapter parameters remain limitations. No new training authorized by this protocol.
