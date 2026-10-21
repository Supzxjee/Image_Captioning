"""Record assistant visual judgments separately from existing human annotations."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Indexed by the locked order, not by a score or a reference caption.
OBSERVATIONS = [
    ('supported_core_relation', '', 'Several people are surfing in the breaking waves.', '', 'People and surfboards.'),
    ('supported_core_relation', '', 'Person visibly holds a surfboard on the beach.', '', 'Person, surfboard and beach.'),
    ('supported_core_relation', '', 'People are seated; the foreground person holds a phone. Bench versus linked chairs is not adjudicated.', '', 'Seated people and a foreground phone.'),
    ('supported_core_relation', '', 'Person is surfing on a wave.', '', 'Person and surfboard.'),
    ('supported_core_relation', '', 'Skateboarder performs a trick at the ramp; exact temporal direction is not inferred.', '', 'Person, skateboard and ramp.'),
    ('supported_core_relation', '', 'Standing person holds an upright surfboard on the sand.', '', 'Person, surfboard and beach.'),
    ('uncertain', 'holding a cell phone', 'A phone is not identifiable in the visible hand; absence of a visible phone does not establish a contradicted relation. Pink shirt and tie are visible.', '', 'Pink shirt and tie.'),
    ('supported_core_relation', '', 'Person and surfboard are in breaking surf; broad surfing description fits.', '', 'Person and surfboard.'),
    ('supported_core_relation', '', 'Surfer is riding a wave; gender is not independently adjudicated.', '', 'Person, board and wave.'),
    ('supported_core_relation', '', 'Surfer and board are airborne during a wave maneuver; this does not contradict broad riding/surfing.', '', 'Surfer and board.'),
    ('object_error', 'on top of a surfboard', 'Visible vehicle is a jet ski with a shaped hull and handlebars, not a surfboard. Do not count category confusion as a pure relation error.', 'a man riding a jet ski on the water.', 'A person riding on the water.'),
    ('supported_core_relation', '', 'Person visibly surfs a wave.', '', 'Person and surfboard.'),
    ('uncertain', 'a group of people riding surfboards on top of a wave', 'Distant surfers are visible; some stand and some paddle. Evidence is insufficient to certify the complete group claim false.', '', 'Several people in the water.'),
    ('supported_core_relation', '', 'Standing person holds a phone in front of the body at a railing.', '', 'Person and phone.'),
    ('supported_core_relation', '', 'Person is on a skateboard at the edge of a concrete obstacle, performing a trick.', '', 'Person, skateboard and concrete obstacle.'),
    ('uncertain', 'talking on a phone', 'Phone is held in front of the face while the person reclines. Speakerphone use cannot be ruled out by a still image.', '', 'Person reclining in bed with a phone.'),
    ('uncertain', 'on skateboards', 'Group contains people holding boards upright and people with feet on boards. A group-level non-universal reading remains possible; do not count as an unambiguous relation error.', '', 'Group of young people with skateboards.'),
    ('supported_core_relation', '', 'Person carries/holds a surfboard while walking in shallow water.', '', 'Person, surfboard and water.'),
    ('uncertain', 'down a flight of stairs', 'Airborne skateboarder is beside stairs; a still frame does not safely establish temporal direction.', '', 'Person, skateboard and stairs.'),
    ('relation_error_candidate', 'sitting on a skateboard', 'Person is crouching on the ground; the skateboard is separate in front of them, with no seated contact on the board. Full image inspected.', 'a boy is crouching beside a skateboard in the street.', 'Person, skateboard and paved ground; street versus skate area remains unverified.'),
    ('supported_core_relation', '', 'People stand on the beach with surfboards; no incompatible relation is asserted.', '', 'Group, beach and surfboards.'),
    ('supported_core_relation', '', 'Person holds a phone to the ear; broad talking-on-phone description fits visible posture.', '', 'Suit, tie and phone.'),
    ('supported_core_relation', '', 'Person visibly surfs a breaking wave.', '', 'Person and surfboard.'),
    ('supported_core_relation', '', 'Person holds a phone against the ear while wearing a tall hat.', '', 'Tall hat and phone.'),
]


def main():
    source = json.loads((HERE/'SCREEN_24_SELECTION.json').read_text(encoding='utf-8'))
    assert len(source['rows']) == len(OBSERVATIONS) == 24
    rows=[]
    for row, observation in zip(source['rows'], OBSERVATIONS):
        status, wrong_span, evidence, correction, protected = observation
        rows.append(dict(**row, assistant_status=status, disputed_span=wrong_span,
            visual_evidence=evidence, minimal_corrected_caption=correction,
            supported_content_to_preserve=protected, human_verified=False,
            protected_contrast_locked=False, decision_tokens_locked=False))
    counts={s:sum(r['assistant_status']==s for r in rows) for s in sorted({r['assistant_status'] for r in rows})}
    assert counts == dict(object_error=1, relation_error_candidate=1, supported_core_relation=17, uncertain=5)
    assert [r['coco_id'] for r in rows if r['assistant_status']=='relation_error_candidate'] == [253341]
    result=dict(status='SCREEN_24_COMPLETE_STOP_LIMIT_REACHED_INSUFFICIENT_RELATION_CASES',
        reviewer_type='assistant_visual_screen_not_human_gold', counts=counts,
        new_relation_error_candidates=[253341], previous_relation_error_candidates=[200984],
        total_relation_error_candidates=2, fully_locked_oracle_cases=0,
        limits=['Targeted 24-image screen, not an estimate of model error prevalence.',
            'Supported core relation does not certify all other caption facts, gender, count, direction or scene nouns.',
            'Protected content is not yet a verified protected contrast.',
            '253341 needs crouching-beside vs sitting-on, which the standing/carrying basis shortlist does not directly cover.',
            '200984 wrong span is to his ear; holding and looking are compatible.',
            'No new model execution, inference, training, or test-set use.'], rows=rows)
    (HERE/'SCREEN_24_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts))


if __name__=='__main__':
    main()
