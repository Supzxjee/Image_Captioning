# FULLFRAME received artifact audit — 21 October 2026

Technical checks PASS: all 10 CPU tests and two-GPU synthetic DDP smoke in run.log; 46/46 raw and quantized feature compatibility comparisons; 300 CE-only updates on each rank; identical initial/final parameter hashes across ranks; original parameters unchanged. All archived source hashes and locked manifest hash match protocol. Before/after records each contain the same 29 unique groups.

This is a completed control arm, not evidence of improved full-caption quality. Teacher-forced ranking on exposed development improves from 3/7 to 6/7 jointly correct groups; on the new probe it improves from 0/8 to 1/8. These scores are not semantic judgments of freely generated phrases. No primary human verdict or crop comparison is available. Do not change the locked crop protocol to improve these observed control results.

Next: preserve the entire `/kaggle/working/pair_crop_shared` folder as a Kaggle dataset, including feature_cache.bin, receipt, images and completed FULLFRAME reference files. Preserve the control adaptation.bin separately. Run the unchanged PAIRCROP notebook at commit36f0145 using the baseline checkpoint and the mounted shared folder. Require matching runtime, source/cache receipts and initial parameter hash. After both runs, conduct blind semantic review under the prespecified decision rule.

The review ZIP does not contain the actual feature cache or adaptation weights; their content hashes cannot be independently recomputed here. PAIRCROP runtime must validate the mounted shared assets. No local model inference or training was run.
