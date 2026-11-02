# Codex1 independent review — 02/11/2026

Reviewed remote main `5e77b01`: visually inspected v3 overview PNG and v2 encoder/decoder PNGs, compared with actual `model.py` encode/Alignment/Fusion/DecoderLayer/decode, and checked source traces against current Git blobs. No model, Torch, Kaggle or PowerPoint execution; native PowerPoint rendering not verified. No figure files edited by Codex1.

## Verdict

**V3 overview: acceptable as an architecture overview for discussion with the supervisor.** Noun/verb frozen banks and learned aggregators are distinct; K150 differs from variable ROI count N; stage1 `(V0,To)→(V1,T1)`, stage2 `(V1,T1)→Vo`, relation `(Vo,Tr)→Vr` and independent Vo/Vr inputs to decoder match code. Frozen raw cache vs learned projection is clear. TRAIN shift-right prefix and VALID generated prefix are separate modes. Valid CIDEr selects checkpoints, not an RL/backprop reward. No unsupported Q-Former/UPT/retrieval or new-method claim. V3 routing is readable in the inspected PNG.

**Encoder detail v2: no material forward discrepancy found.** Cosine/region softmax τ0.1/relevance/prototype weighting, own and cross streams, scalar fusion gate, T1 carry, Vo branch and Vr path agree with code at the displayed level of abstraction.

## Requested decoder-detail revision (v2 slide3)

1. **Show the context bypasses explicitly.** Current drawing connects object context co into gate go, and relation context cr into gate gr, then only gate blocks have outgoing arrows into the sum/FFN box. Its formula `go⊙co + gr⊙cr` is correct, but the data-flow drawing omits co/cr arriving at those products. Please branch each context into both its gate and its multiplication node; gate output goes to the other input of that multiplication. Then sum the two weighted contexts and apply FFN. This is already the implemented decoder; do not change code or propose a new mechanism.
2. **Reroute the H and Vo paths.** The vertical H branch crosses the Vo→object-attention horizontal path without a junction marker/bridge. The intended semantics are Q=H and K,V=Vo, not a join between those streams. Use distinct entry sides/corridors so the crossing disappears. The v3 overview's clean-routing guard does not establish zero crossings in the separate v2 decoder diagram.
3. Optional precision: gate labels may include biases `+bo`/`+br`, since the code uses `nn.Linear(..., bias=True)`. This is a shorthand-label issue, not a wrong gate family or a blocker for the overview.

The overall diagram does not need the VALID teacher-forced NLL branch if it remains an overview: the separate detail already describes it as evaluation only. Do not add reference-prefix arrows into VALID beam generation while clarifying that measurement.

## Artifact checks

- Source hashes in both v3/v2 traces match current Git blobs; model SHA139edbb1… unchanged.
- Actual PPTX XML: v3 has1slide,82native shapes,48connectors,1photo; v2 has3slides with60/53/45shapes and36/29/31connectors,0pictures. Supports editable structure, not proof of native-app appearance or correctness.
- V3 input photo hash matches its trace. It is illustrative input, not measured model output.
- No training result, metric gain or exact-reproduction/novelty claim follows from this drawing review.
