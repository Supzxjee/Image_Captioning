# V4 — parallel conditioning lanes, reroute from scratch

User rejected V3 connector complexity and approved rearranging modules first. Object and relation conditioning are now two horizontal lanes: Slot MLP → masked attention → output projection → residual sum. Query memory runs below the slot stage, key/value enters from above, and the residual uses a separate lane below each attention/projection. Prompt encoding and ROI matching are together above the slots. Bank prototypes stay near the fusion cascade. No unrelated operations are merged.

Main Vo/Vr and Vo′/Vr′ flow remains connected. Auxiliary input tensors I, B and original Vo are repeated near their consumers, explicitly marked as the same tensors. This is diagram notation, not another image encoder/detector/model computation. It avoids long auxiliary cross-figure cables. Both slot MLPs still use original Vo. One actual object fusion is expanded in place; standard Norm/FFN internals are explicitly omitted. Older versions and model/metrics remain unchanged.


## QA / delivery

Font-width checks, unique SVG IDs, 14 required nodes/routes, no operation-node overlap, no connector through operation interiors, and no collinear shared connector sections PASS. These checks do not assert zero perpendicular crossings. Final SVG PNG and Poppler-rendered one-page PDF visually inspected; PDF text selectable. Download copies: Pipeline_song_song_V4_07_11_2026.svg/png/pdf. Main memories remain connected; repeated auxiliary sources explicitly denote identical cached tensors. No model/scorer/notebook or metric/Sheet edits.
