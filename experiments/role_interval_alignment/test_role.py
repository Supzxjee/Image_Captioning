"""Synthetic CPU tests; no torch, spaCy, neural parser or caption-model execution."""
import copy
import unittest
from role_logic import units,evaluate,compare,advantage,identity,select_review
from role_parser import translate
from role_review import audit


def graph(edges,distinct=()):
    cats={'p':'person','q':'person','bat':'baseball bat','horse':'horse','bike':'bicycle'}
    return dict(text='fixture',entities={k:dict(category=cats[k]) for e in edges for k in (e[0],e[2])},edges=[dict(subject=s,predicate=p,object=o) for s,p,o in edges],distinct=list(distinct),opaque=[])


EMPTY=graph([])
HOLD=graph([('p','holding','bat')]);RIDE=graph([('p','riding','horse')])
JOINT=graph([('p','holding','bat'),('p','riding','horse')])
OPAQUE=dict(text='unresolved',entities={},edges=[],distinct=[],opaque=[dict(surface='unresolved',reason='fixture')])
SCENE=dict(instances=[[1,'person',[]],[2,'person',[]],[3,'baseball bat',[]],[4,'horse',[]]],positive_edges=[[1,'holding',3],[2,'riding',4]],actor_labels=[[1,'holding',1],[1,'riding',0],[2,'holding',0],[2,'riding',1]],actor_inventory_complete=True)


def dependency(words,lemmas,pos,deps,heads,tags=None):
    return [dict(text=w,lemma=l,pos=p,dep=d,head=h,tag=t) for w,l,p,d,h,t in zip(words.split(),lemmas.split(),pos.split(),deps.split(),heads,tags or ['NN']*len(heads))]


class Checks(unittest.TestCase):
    def test_joint_actor_not_bag(self):
        u=units(JOINT)[0]
        self.assertEqual(evaluate(u,SCENE)['status'],'refuted')
        self.assertEqual(evaluate(u,SCENE,bag=True)['status'],'supported')
        self.assertTrue(compare(JOINT,EMPTY,SCENE)['binding_operates'])
        s=copy.deepcopy(SCENE);s['positive_edges'].append([1,'riding',4]);s['actor_labels'][1][2]=1
        self.assertEqual(evaluate(u,s)['status'],'supported')

    def test_unknown_not_absence(self):
        for mutation in ('unannotated_actor','no_target_class','incomplete','missing_role'):
            s=copy.deepcopy(SCENE)
            if mutation=='unannotated_actor':s['instances'].append([5,'person',[]])
            if mutation=='no_target_class':s['instances']=[x for x in s['instances'] if x[1]!='horse'];s['positive_edges']=[x for x in s['positive_edges'] if x[2]!=4]
            if mutation=='incomplete':s['actor_inventory_complete']=False
            if mutation=='missing_role':s['actor_labels']=[x for x in s['actor_labels'] if x[:2]!=[1,'riding']]
            self.assertEqual(evaluate(units(JOINT)[0],s)['status'],'unknown',mutation)

    def test_pair_unknown_positive_counterexample(self):
        for a,b in [(OPAQUE,HOLD),(HOLD,OPAQUE)]:
            self.assertEqual(compare(a,b,SCENE)['role_advantage'],0)
        r=compare(OPAQUE,HOLD,SCENE)
        self.assertEqual((r['lower'],r['upper']),(-2,0));self.assertNotEqual(r['unknown_zero_advantage'],0)
        unresolved=copy.deepcopy(OPAQUE);unresolved['opaque'][0]['weight']=3
        self.assertEqual(compare(unresolved,JOINT,SCENE)['role_advantage'],0)

    def test_shared_unknown_cancels(self):
        mixed=copy.deepcopy(HOLD);mixed['opaque']=OPAQUE['opaque']
        r=compare(mixed,OPAQUE,SCENE)
        self.assertEqual(r['role_advantage'],1/3);self.assertEqual(len(r['shared']),1)

    def test_canonical_renaming(self):
        other=copy.deepcopy(JOINT);other['entities']['actor']=other['entities'].pop('p')
        for e in other['edges']:e['subject']='actor'
        self.assertEqual(compare(other,JOINT,SCENE)['role_advantage'],0)
        self.assertEqual(len(compare(other,JOINT,SCENE)['shared']),1)

    def test_distinct_and_cap(self):
        g=graph([('p','holding','bat'),('q','holding','bat')],distinct=[['p','q']])
        self.assertEqual(evaluate(units(g)[0],SCENE)['status'],'refuted')
        g['distinct']=[]
        # Disconnected anonymous mentions can refer to the same known person.
        self.assertTrue(all(evaluate(u,SCENE)['status']=='supported' for u in units(g)))
        for a,b in [(2,1),(float('nan'),1)]:
            with self.assertRaises(ValueError):advantage(a,b)

    def test_dependency_shared_subject_and_conjunction(self):
        ts=dependency('a man holding a baseball bat and riding a horse','a man hold a baseball bat and ride a horse',
            'DET NOUN VERB DET NOUN NOUN CCONJ VERB DET NOUN','det nsubj ROOT det compound dobj cc conj det dobj',[1,2,2,5,5,2,7,2,9,7])
        p=translate('a man holding a baseball bat and riding a horse',ts)
        self.assertFalse(p['opaque']);self.assertEqual(len(p['edges']),2)
        self.assertEqual(p['edges'][0]['subject'],p['edges'][1]['subject'])
        self.assertEqual(evaluate(units(p)[0],SCENE)['status'],'refuted')

    def test_parser_uncertainty_retained(self):
        base=dependency('a person holding a horse','a person hold a horse','DET NOUN VERB DET NOUN','det nsubj ROOT det dobj',[1,2,2,4,2])
        for mutation in ('plural','passive','unknown_target','unresolved_pronoun','synonym','grip','modal','disjunction'):
            ts=copy.deepcopy(base)
            if mutation=='plural':ts[1]['tag']='NNS'
            if mutation=='passive':ts[1]['dep']='nsubjpass'
            if mutation=='unknown_target':ts[4]['text']='thing'
            if mutation=='unresolved_pronoun':ts[1].update(text='they',pos='PRON')
            if mutation=='synonym':ts[2].update(text='carrying',lemma='carry')
            if mutation=='grip':ts[2].update(text='gripping',lemma='grip')
            if mutation=='modal':ts[0]['text']='might'
            if mutation=='disjunction':ts[0]['text']='or'
            p=translate('fixture',ts)
            self.assertTrue(p['opaque'],mutation);self.assertFalse(p['edges'])
            self.assertEqual(p['opaque'][0]['weight'],3)

    def test_duplicate_edges_do_not_inflate_weight(self):
        g=copy.deepcopy(HOLD);g['edges']*=3
        self.assertEqual(units(g),units(HOLD))

    def test_binding_comparator_keeps_negative_label_policy(self):
        s=copy.deepcopy(SCENE);s['positive_edges']=[]
        s['actor_labels']=[[a,p,0] for a in (1,2) for p in ('holding','riding')]
        r=compare(HOLD,EMPTY,s)
        self.assertLess(r['role_advantage'],0);self.assertFalse(r['binding_operates'])

    def test_binding_cap_cannot_create_refutation(self):
        g=graph([('p','holding','bat'),('q','holding','bat')])
        g['entities']['r']=dict(category='person');g['edges'].append(dict(subject='r',predicate='holding',object='bat'))
        s=dict(instances=[[i,'person',[]] for i in range(1,31)]+[[99,'baseball bat',[]]],positive_edges=[],
               actor_labels=[[i,'holding',0] for i in range(1,31)],actor_inventory_complete=True)
        result=evaluate(units(g)[0],s)
        self.assertEqual(result['status'],'unknown');self.assertEqual(result['reason'],'binding_cap')
        self.assertTrue(compare(g,EMPTY,s)['search_capped']);self.assertFalse(compare(g,EMPTY,s)['operative'])

    def test_spatial_phrase_not_riding_and_no_opaque_escape_bonus(self):
        ts=dependency('a person on a horse','a person on a horse','DET NOUN ADP DET NOUN','det ROOT prep det pobj',[1,1,1,4,2])
        p=translate('a person on a horse',ts)
        self.assertTrue(p['opaque']);self.assertFalse(p['edges'])
        self.assertEqual(compare(p,JOINT,SCENE)['role_advantage'],0)

    def test_mixed_role_clause_does_not_drop_uncertainty(self):
        base=dependency('a person holding a baseball bat','a person hold a baseball bat',
            'DET NOUN VERB DET NOUN NOUN','det nsubj ROOT det compound dobj',[1,2,2,5,5,2])
        p=translate('fixture',base);self.assertFalse(p['opaque']);self.assertEqual(len(p['edges']),1)
        ts=base+dependency('astride a horse','astride a horse','ADP DET NOUN','prep det pobj',[1,8,6])
        self.assertEqual(translate('fixture',ts)['opaque'][0]['reason'],'mixed_unhandled_role_like_phrase')
        ts[6]['text']='on';ts[6]['lemma']='on'
        self.assertEqual(translate('fixture',ts)['opaque'][0]['reason'],'mixed_unhandled_on_phrase')

    def test_html_export_is_unfilled_and_bound(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from role_review import write_review
        scene=copy.deepcopy(SCENE);scene['filename']='one.jpg';scene['instances']=[]
        p=dict(graph=EMPTY,dependency_tokens=[])
        c=dict(coco_id=123,scene=scene,greedy='same',sample='same',parsed=dict(sample=p,greedy=p),comparison=compare(EMPTY,EMPTY,scene))
        fixture_parent=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='role-fixture-',dir=fixture_parent) as t:
            root=Path(t).resolve();self.assertEqual(root.parent,fixture_parent)
            Image.new('RGB',(20,20),'white').save(root/'one.jpg')
            write_review(root/'review',[c],root)
            raw=(root/'review/index.html').read_text(encoding='utf-8')
            self.assertIn('role_interval_verified_review.json',raw)
            self.assertIn('value="">',raw);self.assertNotIn('FINGERPRINT',raw)
            self.assertTrue((root/'review/123.jpg').is_file())

    def test_review_selection_and_binding(self):
        cases=[dict(coco_id=i,comparison=dict(role_advantage=1 if i<18 else 0,operative=i%3==0)) for i in range(128)]
        chosen=select_review(cases);self.assertEqual(len(chosen),30);self.assertEqual(len({c['coco_id'] for c in chosen}),30)
        m=dict(kind='role_interval_derivation_review',cases=chosen)
        review=dict(schema_version=1,kind=m['kind'],manifest_sha256=identity(m),reviewer='Human',reviewer_type='human_self_reported',annotations={str(c['coco_id']):dict(decision='correct',missing_link_as_false=False,parser_uncertainty_as_fact=False,notes='') for c in chosen})
        self.assertTrue(audit(m,review)['human_gate_pass'])
        review['annotations'][str(chosen[0]['coco_id'])]['missing_link_as_false']=True
        self.assertFalse(audit(m,review)['human_gate_pass'])
        review['annotations'][str(chosen[0]['coco_id'])]['missing_link_as_false']=False
        review['annotations'][str(chosen[0]['coco_id'])]['parser_uncertainty_as_fact']=True
        self.assertFalse(audit(m,review)['human_gate_pass'])
        review['manifest_sha256']='wrong'
        with self.assertRaises(ValueError):audit(m,review)


if __name__=='__main__':unittest.main(verbosity=2)
