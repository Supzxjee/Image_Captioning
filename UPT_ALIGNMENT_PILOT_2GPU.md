# UPT visual-semantic alignment pilot — four arms, two GPUs per arm

Prepared 2026-10-19. **Not yet trained or evaluated on GPU. No new caption-quality result.**

Each notebook runs **one arm on both GPUs using DDP**. Two independent Kaggle notebooks/accounts can run A/B concurrently; consider C/D only after reviewing A/B. This is not one arm per GPU.

| Arm | Change | Standalone notebook |
|---|---|---|
| A | Continue best SCST + CE + object-reward baseline | [A — two GPUs](experiments/upt_alignment_pilot/UPT_ALIGNMENT_A_2GPU.ipynb) |
| B | Add UPT-selected visual regions, without new UPT text | [B — two GPUs](experiments/upt_alignment_pilot/UPT_ALIGNMENT_B_2GPU.ipynb) |
| C | Add the same regions with direct CLIP text fusion | [C — two GPUs](experiments/upt_alignment_pilot/UPT_ALIGNMENT_C_2GPU.ipynb) |
| D | Text-conditioned visual gates for subject/object/relation slots | [D — two GPUs](experiments/upt_alignment_pilot/UPT_ALIGNMENT_D_2GPU.ipynb) |

Import the notebook, select GPU T4 x2, enable Internet, verify the explicit input mounts and **Save & Run All**. Each notebook embeds its new source; no additional Python file upload is needed. The runtime base repository is intentionally pinned to `1dd4dc15f42eded978945a3da66c65ec6a725b59`; the embedded overlay defines this pilot.

Same deterministic Karpathy train10k/val1000 cohort and preserved checkpoint across all arms. Global batch8 = 4/GPU, 1250 updates; first200 CE-only, then `L = CE + .05*SCST`, reward `CIDEr-D + .5*object_count_v2`. No V-COCO relation supervision or test-based selection. Keep the original legacy prompt cache alongside the new UPT JSON.

Validation uses raw greedy decoding for both the starting and end checkpoints; these screening metrics must not be compared directly to historical beam/reranked test scores. B/C/D also include branch-removal and text/region-swap diagnostics. Gates learn caption utility, not calibrated relation truth; improvement and novelty remain unproven.

Local evidence: three data unit tests passed; syntax and embedded source identities checked in all four unexecuted notebooks; boxes/instance links checked on all11000 selected entries in the real full prompt JSON. GPU model tests and a two-rank DDP smoke test run on Kaggle before expensive preparation/training. No notebook was launched by the assistant.

See [full Vietnamese instructions and limitations](experiments/upt_alignment_pilot/README.md), [local input check](experiments/upt_alignment_pilot/local_input_check.json), and [artifact validation](experiments/upt_alignment_pilot/artifact_validation.json). Return the small review ZIP; preserve the non-ZIP `.pth` checkpoint separately in saved Output.
