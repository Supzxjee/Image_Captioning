import unittest
from probe_data import bounds, vcoco_pairs, human_pairs, summarize


def phrase(verb, obj): return f'a person {verb} a {obj}'


def image(name, verb, negative=0):
    return dict(image_id=int(name), filename=name, instances=[
        dict(instance_id=1,category='person',box=[0,0,.4,1]),
        dict(instance_id=2,category='horse',box=[0,.4,1,1]),
        dict(instance_id=3,category='person',box=[.6,0,1,1])],
        relations=[dict(subject_id=1,object_id=2,predicate=verb)],
        actor_action_labels=[dict(subject_id=1,predicate=verb,label=1),
            dict(subject_id=1,predicate='riding' if verb=='holding' else 'holding',label=negative),
            dict(subject_id=3,predicate=verb,label=negative)])


class ProbeDataChecks(unittest.TestCase):
    def test_no_missing_or_both_positive_as_negative(self):
        images=[image('1','holding',1), image('2','riding',1)]
        captions={r['filename']:dict(split='train') for r in images}
        self.assertEqual(vcoco_pairs(images,set(),captions,phrase),[])
        for row in images: row['actor_action_labels']=row['actor_action_labels'][:1]
        self.assertEqual(vcoco_pairs(images,set(),captions,phrase),[])

    def test_balance_same_class_and_exclude_training(self):
        images=[image('1','holding'),image('2','riding'),image('3','holding')]
        captions={r['filename']:dict(split='train') for r in images}
        rows=vcoco_pairs(images,{'3'},captions,phrase)
        verbs=[r for r in rows if r['task']=='wrong_predicate']
        self.assertEqual(len(verbs),2)
        self.assertEqual({r['predicate'] for r in verbs},{'holding','riding'})
        self.assertTrue(all(r['filename']!='3' for r in rows))
        self.assertTrue(all(r['candidates'][0]['bounds']==r['candidates'][1]['bounds'] for r in verbs))

    def test_human_requires_explicit_clear_contrast(self):
        case=dict(case_id='x', filename='1', subject_box=[0,0,.4,1], relation='holding',
                  targets=[dict(label='horse',bbox=[0,0,.5,1]),dict(label='horse',bbox=[.5,0,1,1])])
        a=dict(subject_box_status='clear',A_box_status='clear',B_box_status='clear',A_status='supported',B_status='contradicted')
        args=({'cases':[case]}, {'annotations':{'x':a}}, set(), {'1':{'split':'train'}}, phrase)
        self.assertEqual(len(human_pairs(*args)),1)
        a['B_status']='supported'
        self.assertEqual(human_pairs(*args),[])
        a['B_status']='contradicted';a['A_box_status']='incorrect'
        self.assertEqual(human_pairs(*args),[])

    def test_ties_are_half_credit_not_side_bias(self):
        row=dict(task='wrong_predicate',predicate='holding',coco_id=1,positive_index=1,
                 scores={k:[.2,.2] for k in ('boxed','global_view','zero_visual')})
        result=summarize([row])['wrong_predicate']['methods']['boxed']
        self.assertEqual(result['pair_accuracy'],.5)
        self.assertEqual(result['ties'],1)

    def test_invalid_box_rejected(self):
        with self.assertRaises(ValueError): bounds([0,0,0,1],[0,0,1,1])


if __name__=='__main__': unittest.main()
