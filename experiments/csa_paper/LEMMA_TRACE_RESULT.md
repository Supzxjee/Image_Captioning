# Exact POS trace — confirmed narrow lemmatization error

02/11/2026. User-run ZIP SHA `0b19128b39de256f5bb1aee0ccbc5624890201efaae369d21d5686bf73c194b2`. Same spaCy3.8.11/Thinc8.3.11/NumPy2.1.3/model3.8.0; all26model-data hashes matched the original126 pipeline per the executed identity guard. Two module/packet/environment receipts match the prepared package. Agent independently recomputed all focus records and aggregate counts from exported tokens, not by running POS locally.

| Focus group in this purposeful sample | Observed mapping | Count |
|---|---|---:|
| TRAIN, riding tagged VERB | riding→rid (expected ride) | 9 |
| TRAIN, rides tagged VERB | rides→rid (expected ride) | 6 |
| Synthetic riding/ride-controls | riding→rid and rides→rid | 2 |
| Synthetic literal rid/ridding | rid→rid, ridding→rid | 2 |

Total45cases:40TRAIN captions/5synthetic controls, not a random sample. TRAIN focus screens:20MATCH/15MISMATCH/5NOT_VERB_REVIEW; synthetic3MATCH/2MISMATCH. No dataset-wide error percentage can be inferred. MATCH only means morphological-screen agreement; e.g. rawTRAIN“ridding skateboards” and“rid inning” are likely caption spelling problems, not certified correct semantics. NOUN“rode” in road-like contexts is left for review, not mapped to ride automatically.

All three previously viewed image captions produced VERB riding→rid. This confirms the preprocessing issue independently of debating whether riding is visually appropriate. Example controlled sentence “A boy is riding a horse.” yielded riding/VERB/VBG/rid. “They rid the room of smoke.” correctly retained rid.

Worker10.965s excludes dependency setup/full notebook wall. No GPU/caption model/CLIP text/optimizer, no bank/ROI modification or Dataset creation. **The observation is confirmed; corrected full-bank counts are not yet measured.**

Decision: apply only VERB +surface riding/rides +originallemma rid→ride, preserve all other tokens. Recount affected TRAIN sentences with exact pipeline and produce a new bank/delta receipt; do not rename every aggregated rid. See [correction protocol](LEMMA_CORRECTION_PROTOCOL.md). Preserve original126/ROI; no full image extraction or fullTRAIN POS rerun. Dataset126 remains unconfirmed until a successful save/path is provided.
