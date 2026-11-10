# Restore CSA structure on cached CLIP patches

10/11/2026. User approved rebuilding the structure described in the preceding CSA review. Budget reported: own account6h, Ngân30h. Author uses ponytail: reuse the existing self-written paper core, leave all historical models/results intact. User runs Kaggle; author executes only metadata/NumPy/AST checks.

## Fixed architecture and disclosed substitutions

Reuse `experiments/csa_paper/model.py` as `bank_layers.py`, with its existing equation/convention audit in `CSA_PAPER_IMPLEMENTATION_SPEC.md`. Raw frozen CLIP-B/16 cache →196patch tokens → Linear768→512 →two complete object Fusion layers →one object-guided relation Fusion layer. Separate noun/verb aggregation (150prototypes each), cosine alignment/weighted pooling, visual self-attention, semantic cross-attention, FFN/postnorm and scalar self/cross gate remain intact. Object2 takes selected semantics from object1. Decoder: three512D layers,8heads,FFN2048,dropout.2; independent object/relation cross-attentions and independent previous-word-conditioned vector gates at EVERY layer. No old strength gate, no old single-memory branch gate, no standard4layer decoder, no Q-Former or caption checkpoint.

Retain the topic's image-specific prompt path as an explicit extension AFTER the paper encoder: object slots use text+pooledVo+box; relation slots use text+subjectVo+objectVo+geometry; masked adapters produce Vo′,Vr′ without fusing the two memories. Reuse the prior zero-initialized PromptMemory with no new loss. Both slot types read the original Vo before either adapter. Empty/unsupported prompt slots contribute exactly zero. This addition is NOT part of original CSA. A tensor may package the two streams by concatenation for existing beam code, but `decode` splits it before TWO independent cross-attentions; packaging is not feature fusion.

This is a paper-structure adaptation plus our prompt extension, NOT exact full-system CSA replication: CLIP patches replace Swin/deformable-DETR regions; CLIP text banks replace RegionCLIP; TRAIN bank lists are our fixed130 cache. Exact detector/RegionCLIP/15XE+30RL reproduction would require different resources and a separate budget decision. No paper-score promise. Two object layers and the dual-memory decoder are restored, not newly proposed mechanisms.

## First runnable notebook

`CLIP_PATCH_CSA_STRUCTURE_SMOKE_2GPU.ipynb`: T4×2, InternetON, four existing inputs visual-cache/mscoco-2014/bank130/prompt175. No API Secret, new semantic cache or old checkpoint. Freshseed42, fixed256TRAINimages,200XEupdates/global32, Adam(.9,.99)/eps1e-8/wd0,clip1/FP32;20step smoke warmup8e-6→1e-4. No VALID/TEST/CIDEr/SCST/reranking. Uses small rawpatch stage (~74MiB), not full33GiB copy. Source and cache seals verified; exact per-image instance/ROI links remapped into fractional14×14patch support. TRAINcaption references only supervise XE.

User-run checks: real two-stream shape and independent decoder response; future-token invariance; empty/invalid-slot identity and active-slot text sensitivity; finite/connected gradients across two object layers/relation/dual decoder/aggregators/adapters; global-token DDP vs full-batch gradient parity; same TRAIN probe NLL decreases; beam5 path scores replay with numerical tolerance, not brittle identical IDs across batching; peak memory and steady speed measured. Dropout OFF for numerical parity, ON for training. Probe does not select architecture/hyperparameters from TEST. Smoke weights are discarded as a basis for full training.

Parent75min stop cap includes setup/staging; prepare35min, worker≤35min with remaining-parent bound. Neither cap is an ETA or GPU billing estimate. Progress heartbeat30s, no auto restart/reinstallTorch. Save source/data receipts, guards,200step logs and result in Working; review ZIP excludes binary model tensors. This is a bounded functional check, not full XE or quality evidence. If it fails/OOM send ZIP instead of raising caps blindly.

## After measured smoke

Full training remains a separate scheduling decision after actual throughput and memory are known. User has not chosen15XE+30SCST; do not interpret rebuilding the method as permission for that compute. Proposed later10XE comparison with baseline199 changes decoder/depth/dropout/encoder together and therefore tests the whole restored architecture, not a clean isolated alignment effect. Do not reuse baseline initialization digest as a requirement for a different decoder. No TEST tuning or unrequested document edits.
