# Training target verification package — R86

Prepared125unknown target alternatives on61training images, including32same-object-class substitutions placed first. No new negative labels have been assigned and no head has been trained from this package.

## Review

Open `http://127.0.0.1:8765/index.html` while the local preview server is running. Offline alternative: extract `C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1_review.zip` and open its `index.html` in a browser. Use one location consistently because browser autosave is tied to the origin.

1. Enter a reviewer name/code.
2. Inspect green subject S and blue targetA/orange targetB, with full scene and two zoomed pair views.
3. Assess subject/target localization: clear, incorrect or unclear. Assess each relation independently: supported, contradicted or unclear.
4. Add an evidence note for contradictions. Lack of visible evidence alone is not proof of a negative.
5. Export `target_instance_review.json` and return it. Partial reviews are supported; prioritize the first32same-class cases before extending to the other93cases.

Model scores and original reference labels are hidden from the UI. A/B order is randomized with fixed seed42; selection does not use model predictions. This is source-label-hidden review, not a claim of a fully blinded or independently adjudicated study.

Relation counts:holding55,sitting_on27,riding18,carrying16,eating7,talking_on2. Images are fetched individually from the official [COCO dataset](https://cocodataset.org/#download) image host; JPEG decoding and dimensions match the preceding Kaggle extraction metadata. Image URLs/hashes are retained locally. Byte identity with the Kaggle dataset files has not been established.

Public review ZIP is approximately10.6MB and contains only the UI, public cases and61images. No checkpoint/GPU/Kaggle run is needed. Private positive-side mapping and score histories are omitted from the public review folder/ZIP. No images are committed to GitHub.

## Import rules

`import_target_annotations.py` requires the exact manifest, unchanged training pair source, a named self-reported human reviewer, valid case IDs and label values. It filters eligible contrasts only when:

- Subject and both target boxes are clearly localized with correct object classes.
- Original positive-side target is independently marked supported.
- The alternative target is marked contradicted with an evidence note.

Incomplete, unclear, wrong-box, both-supported, source-positive-conflicting and missing-evidence cases remain excluded. Neither original training labels nor model weights are modified. An alternative marked supported is not fabricated into a negative; a contradiction of the source positive requires separate resolution rather than silent relabelling.

Reviewer identity is self-reported, not a certification of independence/consensus. Explicit verified target-ranking training and a no-target-loss control remain the next stage **after eligible annotations exist**. The inspected full100 evaluation examples are not used as training labels here.

## Validation

Three importer tests pass: blank/unknown remains unlabelled; eligible contrasts require good boxes/evidence and source consistency; wrong manifest/assistant provenance is rejected. Browser QA in an isolated headless session confirms125cases render, next/previous navigation, autosave/reload andJSON export. QA uses unclear statuses only; its export yields0eligiblecontrasts and is kept separate from real labels. Screenshots were inspected for image and box layout.

The in-app computer-use runtime failed to initialize; browser QA therefore used a separate headless browser against the locally authored page. It did not interact with the user's real browser profile. The app preview was requested separately.

## Reproduce

Prepare with `prepare_target_annotation.py --output <package directory> --region-directory <R83 paired output>`. It derives the125cases from unchanged detector-training labels and verifies61image sizes against real extraction records.

Import a returned review with `import_target_annotations.py --review <human JSON> --package <original package directory> --output <new eligible-contrast JSON>`. This command only filters labels; it does not train.

Local package: `C:/KhoaLuanTotNghiep/Code/output/target_swap_annotation_v1`.

Manifest SHA256: `f009ad534f2c24033bb38f363662ce9012350229f48f2ff531b6733527f7ed96`.

Status: ready for human verification;0human annotations received,0negatives assigned,0target-loss training runs. No accuracy, caption-quality or scientific-novelty claim is made.
