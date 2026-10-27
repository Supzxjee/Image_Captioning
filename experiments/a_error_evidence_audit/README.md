# A error/evidence atlas — no new GPU run

Read RESULT.md first. This is a diagnostic review, not a trained method, a new
benchmark,20 confirmed errors or a representative accuracy estimate.

- `cases20.json`:20 focus cases, saved branch captions, references and explicit
  assistant observations/uncertainties.9 clear errors,4 unsupported claims,
  6 under-specified but potentially correct captions,1 uncertain fine-detail case.
- `controls10.json`:5 acceptable captions and5 uncertain cases retained separately.
- `visual_notes.json`: all30 assistant judgments, not human gold/training labels.
- `ARGON_BRAINSTORM.md`: standalone context and request for an independent critique.
- `review/index.html`: local offline viewer; source images and224px human previews.
  `review/controls.html` retains controls. Downloaded images/caches stay ignored.

Rebuild on the research workspace, using existing cached artifacts:

```
python experiments/a_error_evidence_audit/build_packet.py
python experiments/a_error_evidence_audit/fetch_images.py
python experiments/a_error_evidence_audit/finish_packet.py
python experiments/a_error_evidence_audit/test_packet.py
```

The image step retrieves at most30 selected public COCO images, checks decoding
and records bytes/SHA/source URL. It loads no neural model.224px previews are not
the exact model cache input and say nothing about model cue discrimination.

Original prompt text, per-case CLIP/Q tensors and UPT hidden/native-role outputs
are not present in this atlas. Cached protocols support input-source/cohort
coverage only. Without additional controlled observations we cannot attribute
the errors uniquely to visual encoding, alignment or decoder non-use.
