"""Strict teacher-output parsing. A VLM decision is a proposal, never ground truth."""
import json

ALLOWED_RELATIONS = frozenset({
    'holding', 'wearing', 'carrying', 'riding', 'eating', 'looking_at', 'talking_on', 'using',
    'sitting_on', 'standing_on', 'kneeling_on', 'standing_beside', 'sitting_beside',
    'kneeling_beside', 'near',
})


def audit_prompt(pair):
    """No old heuristic predicate, CLIP score or reference caption is disclosed."""
    return (
        'Inspect ONLY the specific pair marked in these images. Image 1 is full context: '
        'red rectangle A is the subject, blue rectangle B is the object. Image 2 is a '
        'closer crop of that same pair, with the same marks. Image 3 is an unmarked detail '
        'of object B. Do not use another person or another object in the picture. '
        f'Detector guesses: A={pair["subject"]}, B={pair["object"]}. These guesses may be wrong. '
        'Identify A and B visually first; reject a wrong guessed class. '
        'Describe the actual visible interaction. Do not infer riding merely from a person '
        'near a board, or infer holding merely from proximity. A person may hold AND wear '
        'a glove; return multiple relations when they are visibly supported. A seated '
        'person wearing a tie is not sitting beside the tie. '
        'If tiny, occluded, ambiguous, or no visible interaction, abstain. '
        'Allowed relation codes: ' + ', '.join(sorted(ALLOWED_RELATIONS)) + '. '
        'Return ONLY one JSON object with these exact fields: '
        '{"subject_visible":true,"object_visible":true, '
        '"subject_description":"visible identity of A", '
        '"object_description":"visible identity of B", '
        '"subject_label_status":"supported|mismatch|unclear", '
        '"object_label_status":"supported|mismatch|unclear", '
        '"relation_status":"supported|none|unclear", '
        '"relations":["allowed_code"], '
        '"uncertainty":"none|small|occluded|instance_ambiguous|object_unclear|action_unclear", '
        '"evidence":"at most 30 words describing visible contact/pose/location"}. '
        'For none/unclear relation_status use an empty relations list. Do not output a '
        'confidence probability. Evidence must describe observable details, not reasoning steps.'
    )


def parse_teacher_answer(text, repair_status=False):
    start = text.find('{')
    if start < 0:
        raise ValueError('No JSON object.')
    answer, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(answer, dict):
        raise ValueError('Expected a JSON object.')
    # Only repair the exact redundant-predicate schema error; never infer a label.
    status = answer.get('relation_status')
    if (repair_status and isinstance(status, str) and status in ALLOWED_RELATIONS
            and answer.get('relations') == [status]):
        answer['relation_status'] = 'supported'
        answer['schema_repair'] = {'field': 'relation_status', 'original': status,
                                   'replacement': 'supported'}
    for key in ('subject_visible', 'object_visible'):
        if type(answer.get(key)) is not bool:
            raise ValueError(f'{key} must be a JSON boolean.')
    for key in ('subject_description', 'object_description', 'evidence'):
        if not isinstance(answer.get(key), str) or not answer[key].strip():
            raise ValueError(f'Missing {key}.')
    if answer.get('object_label_status') not in {'supported', 'mismatch', 'unclear'}:
        raise ValueError('Invalid object-label status.')
    if answer.get('subject_label_status') not in {'supported', 'mismatch', 'unclear'}:
        raise ValueError('Invalid subject-label status.')
    if answer.get('relation_status') not in {'supported', 'none', 'unclear'}:
        raise ValueError('Invalid relation status.')
    if answer.get('uncertainty') not in {
        'none', 'small', 'occluded', 'instance_ambiguous', 'object_unclear', 'action_unclear',
    }:
        raise ValueError('Invalid uncertainty tag.')
    relations = answer.get('relations')
    if not isinstance(relations, list) or any(not isinstance(x, str) or x not in ALLOWED_RELATIONS
                                            for x in relations):
        raise ValueError('Unknown relation code.')
    if answer['relation_status'] == 'supported' and not relations:
        raise ValueError('Supported status with no relations.')
    if answer['relation_status'] != 'supported' and relations:
        raise ValueError('Non-supported status with relations.')
    answer['relations'] = list(dict.fromkeys(relations))
    if answer['subject_label_status'] == 'mismatch':
        decision = 'subject_label_mismatch'
    elif answer['object_label_status'] == 'mismatch':
        decision = 'object_label_mismatch'
    elif not answer['subject_visible'] or not answer['object_visible']:
        decision = 'not_visible'
    elif (answer['object_label_status'] != 'supported' or answer['subject_label_status'] != 'supported'
          or answer['uncertainty'] != 'none'):
        decision = 'unclear'
    elif answer['relation_status'] == 'none':
        decision = 'no_interaction'
    elif answer['relation_status'] == 'unclear':
        decision = 'unclear'
    else:
        decision = 'proposal_for_review'
    return {**answer, 'decision': decision,
            'proposed_relations': answer['relations'] if decision == 'proposal_for_review' else []}


def shard_pairs(pairs, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid shard.')
    return [(i, pair) for i, pair in enumerate(pairs) if i % count == index]


def verification_prompt(pair, predicate):
    if predicate not in ALLOWED_RELATIONS:
        raise ValueError('Unknown claim predicate.')
    return (
        'Verify ONE claim from visual evidence, independently of any previous answer. '
        'Image 1: full image, red A is the subject and blue B is the object. '
        'Image 2: marked crop of the same pair. Image 3: unmarked detail of B. '
        f'Detector guesses A={pair["subject"]}, B={pair["object"]}; verify their identities. '
        f'Claim: marked A {predicate.replace("_", " ")} marked B. '
        'Check the exact marked instances, not another person or nearby item. '
        'Describe A and B first. Then report the visible physical/pose/location evidence '
        'for this claim involving exactly B. For holding/carrying require actual grip or '
        'support; proximity is insufficient. Wearing requires B attached to A, not other '
        'clothing. Sitting_on requires A supported by B, not an unrelated seat above B. '
        'Eating requires visible consumption, not merely holding food. '
        'If evidence actually refers to another actor/object, explicitly flag it. '
        'If hidden, tiny or ambiguous, abstain. Do not substitute another relation. '
        'Return ONLY JSON: {"subject_identity":"supported|mismatch|unclear", '
        '"object_identity":"supported|mismatch|unclear", '
        '"subject_description":"identity of marked A", "object_description":"identity of B", '
        '"actor_is_marked_A":true,"target_is_marked_B":true, '
        '"other_actor_or_object_explains_evidence":false, '
        '"direct_evidence":"observable evidence involving A and B, or not visible", '
        '"verdict":"supported|contradicted|unclear"}. '
        'Booleans must be JSON booleans. Do not give a confidence score.'
    )


def parse_verification(text):
    start = text.find('{')
    if start < 0:
        raise ValueError('No JSON object.')
    answer, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(answer, dict):
        raise ValueError('Expected a JSON object.')
    for key in ('subject_identity', 'object_identity'):
        if answer.get(key) not in {'supported', 'mismatch', 'unclear'}:
            raise ValueError(f'Invalid {key}.')
    for key in ('actor_is_marked_A', 'target_is_marked_B',
                'other_actor_or_object_explains_evidence'):
        if type(answer.get(key)) is not bool:
            raise ValueError(f'{key} must be a JSON boolean.')
    for key in ('subject_description', 'object_description', 'direct_evidence'):
        if not isinstance(answer.get(key), str) or not answer[key].strip():
            raise ValueError(f'Missing {key}.')
    if answer.get('verdict') not in {'supported', 'contradicted', 'unclear'}:
        raise ValueError('Invalid verdict.')
    eligible = (answer['verdict'] == 'supported'
                and answer['subject_identity'] == answer['object_identity'] == 'supported'
                and answer['actor_is_marked_A'] and answer['target_is_marked_B']
                and not answer['other_actor_or_object_explains_evidence'])
    return {**answer, 'eligible_for_review': eligible}
