"""Pure token-record inspection, no model imports and no token/lemma mutation."""
from collections import Counter

EXPECTED = dict(riding='ride', rides='ride', ride='ride', ridden='ride', rode='ride', rid='rid', ridding='rid')


def inspect_doc(doc):
    tokens, focuses = [], []
    for index, token in enumerate(doc):
        record = dict(index=index, surface=token.text, pos=token.pos_, tag=token.tag_,
                      morph=str(token.morph), lemma=token.lemma_)
        tokens.append(record)
        expected = EXPECTED.get(token.text.lower())
        if expected is not None:
            status = 'NOT_VERB_REVIEW' if token.pos_ != 'VERB' else 'MATCH' if token.lemma_.lower() == expected else 'MISMATCH'
            focuses.append(dict(record, expected_lemma_for_verb=expected, status=status))
    return dict(tokens=tokens, focus=focuses)


def summarize(records):
    counts = Counter()
    transitions = Counter()
    for record in records:
        for focus in record['focus']:
            counts[(record['kind'], focus['status'])] += 1
            transitions[(record['kind'], focus['surface'].lower(), focus['pos'], focus['lemma'])] += 1
    return dict(case_count=len(records), focus_counts=[dict(kind=kind, status=status, count=count)
            for (kind, status), count in sorted(counts.items())],
        transitions=[dict(kind=kind, surface=surface, pos=pos, lemma=lemma, count=count)
            for (kind, surface, pos, lemma), count in sorted(transitions.items())],
        bank_modified=False, dataset_error_rate_estimated=False)
