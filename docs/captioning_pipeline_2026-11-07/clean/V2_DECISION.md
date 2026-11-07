# V2 — mechanism at its point of use

User approved changing the layout again after comparing the sparse draft to the reference. Expand one real instance (Object fusion 2) in place, instead of a detached shared inset. Keep Object fusion 1 and Relation fusion as actual separate stages and state that they use the same internal structure with separate parameters. Put the two slot consumers directly above/below their common slot-building location, so prompt text and matching do not travel around the whole figure. Keep decoder and selection at the right. Remove duplicated prose and typography that looks like code. No model/metric changes; earlier drafts remain available.

The expanded instance must include re-selection of T1, self-attention, cross-attention, gate, and H bypass, with Q/K/V indicated. Original Vo supplies both relation stage and slot inputs. Combining slot builders in a single drawing means separate object/relation computations, not shared weights or merged memories. Norm/FFN are compressed in the expanded block with a note. Text/metadata buses represent assembled inputs, not a new learned module. Output: pipeline-clean-v2.svg/png/pdf/html.


## Complete

SVG/PNG/PDF one-page final exports reviewed. Approximate Arial box-width, unique IDs and 13 required data-flow routes PASS. PDF selectable text and one page checked. The long verb prototype route stays outside operation boxes and headings. Earlier drafts preserved. Downloads Pipeline_clean_V2_07_11_2026.*. No model/Kaggle/scorer or result/Sheet edits.
