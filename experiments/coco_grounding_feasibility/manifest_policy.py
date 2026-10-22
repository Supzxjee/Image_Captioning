"""Frozen conservative policy before the full-annotation coverage analysis."""
from audit import AMBIGUOUS

POLICY=dict(version='coco-noun-box-v1',ambiguity_exclusions=sorted(AMBIGUOUS),
    compound_suffixes='box boxes bag bags case cases cover covers wrapper wrappers container containers holder holders rack racks shop shops store stores costume costumes logo logos sign signs poster posters picture pictures photo photos photograph photographs drawing drawings painting paintings sculpture sculptures statue statues toy toys figurine figurines model models food foods collar collars leash leashes bed beds bowl bowls house houses'.split(),
    artificial_prefixes=['toy','fake','model','costume'],
    depiction_cues=['on a poster','on the poster','on a sign','on the sign','on a screen','on the screen',
        'on a television','on the television','in an advertisement','on an advertisement'],
    count_all_instances_before_crowd_filter=True,instances_per_class=1,crowd_allowed=False,
    minimum_patch_area=4.,maximum_image_fraction=.5,patch_grid=14,max_caption_length=30,
    person_maximum_fraction=.25,person_cap_order='sha256(coco-route-person-cap-v1:span_id)',
    minimum_images=2000,minimum_unique_spans=5000,minimum_classes_with_50_spans=15,
    captions_per_image=5,manual_records=16,
    manual_selection='One hash-first record in each of first16 hash-ordered retained classes; if fewer classes, fill by global record hash.',
    supervision_schedule='All retained spans for each main-caption image across its first5references, sorted by fixed per-image span hash; preserve all10k main-caption images. No minibatch/RNG order claim.',
    source_scope='Exact locked A train10k; never held-out supervision',test_used=False)


def context_exclusion(tokens,start,end,label):
    if end<len(tokens) and tokens[end] in POLICY['compound_suffixes']:
        return 'compound_head_changes_object'
    if start and tokens[start-1] in POLICY['artificial_prefixes'] and label!='teddy bear':
        return 'artificial_object_modifier'
    phrase=' '+' '.join(tokens)+' '
    if any(' '+cue+' ' in phrase for cue in POLICY['depiction_cues']):
        return 'depiction_context'
    return None
