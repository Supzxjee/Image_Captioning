# Direct decoder relation memory: completed 12-image routing probe

The new routing changes returned candidate wording/order on six images, but changes no top caption. Target-box swaps change one image's third candidate, including an unsupported chair claim. No trained adaptation or quality improvement is established.

| Comparison | Top changed | Ordered candidate list changed | Candidate set changed |
|---|---:|---:|---:|
| Legacy on vs off | 0/12 | 0/12 | 0/12 |
| Legacy on vs target swap | 0/12 | 0/12 | 0/12 |
| Direct on vs off | 0/12 | 6/12 | 4/12 |
| Direct on vs target swap | 0/12 | 1/12 | 1/12 |
| Legacy on vs direct on | 0/12 | 6/12 | 4/12 |

Six list changes comprise four substitutions and two reorder-only cases. A list change is not six improved captions. Mean absolute logprob changes for shared caption strings are .059990 (direct off) and .004313 (direct target swap), versus .001148 and .000004263 for legacy. Only shared wording is compared; scores for different sentences are not treated as a matched score delta.

Image329841: original direct third candidate is `a little girl sitting on a floor with a book.`; swapped third candidate is `a little girl sitting on a chair with a book.` The original image was viewed from the prior full100 review archive: the child is seated on a blue floor mat holding a book. The chair wording is unsupported in that image. This is an assistant visual spot-check, not independent human evaluation, and is evaluation-only. The alternate target itself has unknown relation correctness. No top-caption change occurs.

The direct off full-memory norm change averages .180021 (18.0021%), and target swap averages .018317 (1.8317%). These numbers are not percentages of semantic contribution or caption improvement. The off statistic includes zeroing the appended relation slots; simply adding/removing memory content inflates it relative to legacy. Do not compare these magnitudes as evidence of better reasoning or decoder attention allocation.

Summary counts were recomputed from 12 distinct records; all direct/off/swap/routing top-change counts agree. The checkpoint and generic prompt hashes match the previous R91 probe. Grounded-cache file hashes differ, so byte-identical caches are not asserted. Code commit reported by output: cc4dfc6. Review ZIP SHA256: 40dc9c2b879b573fb1128e1fa437d206e587ba9fcb081d6b8b6eb4f5444b3fb2.

Decision: keep this as a working architectural hypothesis, not a promoted model. Stop further annotation expansion and repeated frozen-routing variants. The next useful experiment is small matched caption adaptation of legacy and direct-memory arms from the same checkpoint, with train/evaluation image separation and a training-eligible relation cache. Judge supported caption content as well as sensitivity and language metrics. The existing diagnostic cache is inference-only and must not be passed to training. Do not force the inspected chair case into training or tune weights to this sample.

Limits: 12 selected, previously inspected development images; three beam candidates; untrained new routing; generic relation text remains in every arm; no complete correctness labels, repeated on/on condition, CIDEr/BLEU/CHAIR, or heldout comparison. Caption selection/re-ranking has not been implemented by this probe.

