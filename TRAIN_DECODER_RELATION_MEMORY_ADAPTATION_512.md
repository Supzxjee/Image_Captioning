# Matched decoder relation memory adaptation pilot

R93 changed candidate lists but no top captions and produced an unsupported chair candidate under target swap. The next step is controlled adaptation, not more annotation or gate amplification.

Fixed protocol: 512 action-bearing COCO train/restval images, seed42, excluding all100 inspected selected-pool filenames. Both arms start from the same historical E3 V1 checkpoint. Two epochs each, batch16 images, first-five available captions per image, AdamW lr1e-5/weight_decay.01, clipnorm1. Only decoder cross-attention and norm2 train; encoder and other decoder parameters freeze. Caption CE only. Same cohort/order/parameter names and seed; different memory shapes can produce different dropout masks. No early stopping or evaluation-based tuning.

Training uses the original full V1 cache (instance_pair_grounded_relation_v1), with heuristic relations rather than human ground truth or the newest selector's predictions. Only holding/riding/carrying/wearing slots are active in both arms. The selected inference-only diagnostic cache is rejected for training. This tests decoder routing adaptation, not better prompt generation.

Both arms are evaluated on the same inspected12 development images using selected relations, on/off/target-swap. These images are disjoint from the512 training images, but this is not an untouched heldout scientific test. Report supported content and relation errors as well as sensitivity. No quality or novelty claim is established by this pilot.

The source caption checkpoint may already have seen these COCO training images during baseline training. Only the new adaptation stage excludes them; exclusion is not an independent heldout test.

16 local tests PASS: cached encoder/grounding, direct-memory checks, deterministic disjoint cohort selection, unchanged source masks, parameter scope and a finite synthetic optimizer step. The real512-image pilot has not run locally. Syntax checked.

Import DECODER_RELATION_MEMORY_ADAPTATION_512_FULL.ipynb. It is standalone. Fill SELECTED with the same mounted JSON path as the previous run. Checkpoint, prompt cache, full training grounded cache, visual cache and COCO_JSON have the previously supplied paths; attach those inputs if absent. COCO_JSON supplies captions; no COCO image files are read. No full visual-cache staging. Enable Internet and GPU; one GPU is used sequentially.

Code pinned to ef772a1. Return decoder_relation_memory_adaptation_512_review.zip (JSON evidence only). Both legacy/direct adapted_epoch_2.pth checkpoints remain separately in saved Kaggle output, excluded from the review ZIP.
