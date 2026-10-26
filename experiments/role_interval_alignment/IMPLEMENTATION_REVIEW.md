# Static review and validation — 2026-10-26

User approved feasibility packaging only. No local neural parser/caption model or
Kaggle inference/training was launched by the implementation agent.

Astra reviewed the five new runtime/test files in thread
`01a15544-6a7a-7903-a6e1-23e7b9f6caa6`:

- Initial review turn `01a17849-b0ba-7743-b0fd-e8b419772b26` identified mixed
  unsupported role clauses, packet provenance, parser false-certainty critical
  flags and cap-induced apparent binding effects. Implementation fixed all four.
- Follow-up turn `01a17851-27ff-78c1-8412-5aece7c708f2` found no remaining known
  logic blocker within those four fixes. This is **static review**, not a full
  packet PASS, parser accuracy claim or training approval.
- The follow-up correctly found a weak regression: bare `bat` was an excluded
  alias, so the mixed-on fixture could pass for the wrong reason. It now uses
  `baseball bat`, verifies the baseline parses one edge, and asserts the exact
  opaque reason for both mixed astride and mixed on mutations.

Additional root fixes: deduplicate graph edges; bag comparator keeps the same
explicit-negative/unknown evaluator while dropping only joint identity;
whole-caption opaque has weight3 and parser caps the projected caption at3edges.
This closes the concrete false-weight2→opaque-weight1 escape counterexample.
Finite parser coverage remains a limitation, not a universal anti-gaming proof.

See VALIDATION.json for CPU metadata and synthetic-test results. Actual cache
coverage, GPU runtime, generated-caption parsing, density and human audit remain
pending the user's run. There is no new caption score, improvement or established
paper contribution from this preparation.
