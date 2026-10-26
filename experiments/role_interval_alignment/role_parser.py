"""Conservative dependency-to-role projection. No NLP model is loaded on import.

The small fixed spaCy model is invoked only by the user-run Kaggle finish step.
This is a coarse class projection, not a complete caption truth checker.
"""
import re

MODEL_SHA='1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'
MODEL_URL='https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
PARSER_POLICY=dict(spacy='3.8.7',model='en_core_web_sm',model_version='3.8.0',wheel_sha256=MODEL_SHA,
    actions={'hold':'holding','ride':'riding'},projection='singular, positive active hold/ride; coarse COCO classes',
    uncertainty='whole role projection opaque if any potentially scoped clause cannot be resolved',
    unsupported_scope_cues=['carry','carrying','carried','straddle','straddling','mounted','grip','gripping','clutch','clutching','grasp','grasping'],
    no_claim='No gender, adjective, complete-caption truth, or selected-instance verification.')


def translate(text,tokens):
    from captioning.object_consistency import ALIASES
    words=[t['text'].lower() for t in tokens];entities={};edges=[];distinct=[];notes=[]
    children={i:[j for j,t in enumerate(tokens) if t['head']==i and j!=i] for i in range(len(tokens))}
    aliases={};occupied=set()
    excluded={'vehicle','vehicles','ball','balls','bat','bats','plant','plants','table','tables','monitor','monitors','glove','gloves'}
    for phrase,category in sorted(ALIASES,key=lambda p:(-len(p[0]),p[0])):
        for i in range(len(tokens)-len(phrase)+1):
            span=set(range(i,i+len(phrase)))
            if tuple(words[i:i+len(phrase)])!=phrase or span&occupied or ' '.join(phrase) in excluded:continue
            roots=[j for j in span if tokens[j]['head'] not in span or tokens[j]['head']==j]
            if len(roots)!=1:continue
            aliases[roots[0]]=(category,' '.join(phrase),span);occupied|=span
    def opaque(reason):
        # Entire projected caption has at most three edges. An unresolved parse must
        # retain that full bound; a smaller bound could reward escaping a false graph.
        return dict(text=text,entities={},edges=[],distinct=[],opaque=[dict(surface=' '.join(words),reason=reason,weight=3)],notes=notes)
    actions=[i for i,t in enumerate(tokens) if t['lemma'].lower() in PARSER_POLICY['actions']]
    synonyms={'carry','straddle','mount','grip','clutch','grasp'}
    potential=actions or any(t['lemma'].lower() in synonyms or t['text'].lower() in PARSER_POLICY['unsupported_scope_cues'] for t in tokens)
    spatial_candidate=any(w in {'on','with','atop','astride'} for w in words) and any(c=='person' for c,_,_ in aliases.values()) and len(aliases)>1
    potential=potential or spatial_candidate
    if not potential:return dict(text=text,entities={},edges=[],distinct=[],opaque=[],notes=['No in-scope proposition.'])
    if any(t['lemma'].lower() in synonyms for t in tokens):return opaque('potentially_scoped_synonym_not_equated_to_hold_or_ride')
    if not actions:return opaque('potential_role_like_spatial_phrase_not_equated_to_action')
    if any(w in {'with','astride','atop'} for w in words):return opaque('mixed_unhandled_role_like_phrase')
    if any(w in {'photo','photograph','picture','painting','statue','toy','drawing','poster'} for w in words):return opaque('depiction_or_toy_context')
    if any(w in {'if','whether','might','may','could','would','should','will','can','perhaps','maybe','apparently'} for w in words):return opaque('modal_or_nonfactual_context')
    if any(w in {'or','nor','before','after','until','unless'} for w in words):return opaque('disjunction_or_temporal_condition')
    if any(t['lemma'].lower() in {'try','pretend','want','wish','plan','seem','appear','stop','start','begin','finish'} for t in tokens):return opaque('aspect_or_intensional_context')
    if any(t['dep']=='neg' or t['lemma'].lower() in {'not','never','without','no'} for t in tokens):return opaque('negated_or_negative_context')
    if any(t['dep'] in {'nsubjpass','auxpass','csubjpass'} for t in tokens):return opaque('passive_not_supported')
    def resolve(i,trail=()):
        if i in trail:raise ValueError('cyclic_coreference')
        t=tokens[i]
        if words[i] in {'who','that'} and t['dep'] in {'nsubj','dobj','obj'}:
            verb=t['head'];head=tokens[verb]['head']
            if tokens[verb]['dep']=='relcl':return resolve(head,trail+(i,))
        if words[i] in {'he','she'}:
            prior=[k for k,(c,_,_) in aliases.items() if k<i and c=='person']
            if len(prior)==1:return resolve(prior[0],trail+(i,))
            raise ValueError('ambiguous_pronoun')
        if i not in aliases:raise ValueError('unknown_entity_or_pronoun')
        category,surface,span=aliases[i]
        modifiers=[tokens[k] for k in children[i] if k not in span]
        if t.get('tag') in {'NNS','NNPS'} and category not in {'skis','scissors'}:raise ValueError('plural_cardinality_not_modeled')
        if any(x['dep']=='nummod' and x['lemma'] not in {'one','1'} for x in modifiers):raise ValueError('cardinality_not_modeled')
        if any(x['text'].lower() in {'left','right','first','second','last','each','every','all','both','some','several','many','few','neither','either'} for x in modifiers):raise ValueError('identifying_or_quantified_modifier')
        key='n'+str(i)
        if key not in entities:
            entities[key]=dict(category=category,surface=surface,projection_notes=['Gender/attributes not verified.'] if surface!=category else [])
            if any(x['lemma'].lower() in {'another','other','different'} for x in modifiers):
                previous=[k for k,v in entities.items() if k!=key and v['category']==category]
                if len(previous)!=1:raise ValueError('unresolved_distinct_antecedent')
                distinct.append([previous[0],key])
        return key
    def subject(i,trail=()):
        if i in trail:raise ValueError('cyclic_subject')
        found=[k for k in children[i] if tokens[k]['dep']=='nsubj']
        if len(found)==1:return found[0]
        if found:raise ValueError('multiple_subjects')
        t=tokens[i]
        if t['dep']=='conj':return subject(t['head'],trail+(i,))
        if t['dep'] in {'acl','relcl'} and tokens[t['head']]['pos'] in {'NOUN','PROPN'}:return t['head']
        raise ValueError('unresolved_subject')
    def conjuncts(i):
        result=[i]
        for k in children[i]:
            if tokens[k]['dep']=='conj':result+=conjuncts(k)
        return result
    try:
        for i in actions:
            t=tokens[i]
            if t['pos']!='VERB':raise ValueError('action_not_parsed_as_verb')
            sub=subject(i)
            if len(conjuncts(sub))!=1:raise ValueError('coordinated_subject_cardinality')
            actor=resolve(sub)
            if entities[actor]['category']!='person':raise ValueError('nonperson_subject')
            targets=[k for k in children[i] if tokens[k]['dep'] in {'dobj','obj'}]
            if not targets and t['lemma']=='ride':
                targets=[k for p in children[i] if words[p]=='on' and tokens[p]['dep']=='prep' for k in children[p] if tokens[k]['dep']=='pobj']
            if len(targets)!=1:raise ValueError('missing_or_ambiguous_target')
            for obj in conjuncts(targets[0]):
                target=resolve(obj)
                if target==actor:raise ValueError('self_role_not_supported')
                edge=dict(subject=actor,predicate=PARSER_POLICY['actions'][t['lemma']],object=target)
                if edge not in edges:edges.append(edge)
        if not edges:return opaque('potential_action_unresolved')
        if len(edges)>3:return opaque('caption_edge_cap')
        for i,t in enumerate(tokens):
            if words[i]=='on':
                head=t['head']
                if t['dep']!='prep' or tokens[head]['lemma']!='ride' or any(tokens[k]['dep'] in {'dobj','obj'} for k in children[head]):
                    return opaque('mixed_unhandled_on_phrase')
    except ValueError as e:return opaque(str(e))
    return dict(text=text,entities=entities,edges=edges,distinct=distinct,opaque=[],notes=notes)


def parse_caption(nlp,text):
    doc=nlp(text)
    tokens=[dict(text=t.text,lemma=t.lemma_.lower(),pos=t.pos_,tag=t.tag_,dep=t.dep_,head=t.head.i) for t in doc]
    return dict(graph=translate(text,tokens),dependency_tokens=tokens)
