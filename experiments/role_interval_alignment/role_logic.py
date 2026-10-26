"""Entity-consistent partial-label utility. Pure Python; no model imports."""
import hashlib
import itertools
import json
import math

POLICY=dict(version='role-interval-feasibility-v1',seed=42,sample_images=128,review_cases=30,min_pool=500,
    min_active_pairs=16,min_operative_pairs=8,human_correct_required=27,normalizer=3,max_bindings=10000,
    max_component_edges=3,max_component_entities=6,max_preparation_seconds=1800,
    predicates=['holding','riding'],selection='hash-first eligible FIT; one greedy and one multinomial temperature1 sampled caption/image',
    human_critical_flags=['missing_link_as_false','parser_uncertainty_as_fact'],capped_cases_count_toward_operative_gate=False,
    review_selection='10 hash-first active,10 hash-first operative not already selected,10 remaining; fill shortfall hash-first',
    scope='Coarse COCO class role projection; not full-caption truth or selected-instance grounding.')


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def order(value):return hashlib.sha256(('role-interval-fit-v1:'+str(value)).encode()).hexdigest()


def canonical(graph):
    """Small graph isomorphism key; preserve shared entities and explicit inequality."""
    entities=graph['entities'];names=sorted(entities)
    if len(names)>POLICY['max_component_entities']:raise ValueError('Oversized component must be opaque.')
    versions=[]
    for permutation in itertools.permutations(names):
        index={name:i for i,name in enumerate(permutation)}
        versions.append(json.dumps(dict(classes=[entities[n]['category'] for n in permutation],
            edges=sorted([index[e['subject']],e['predicate'],index[e['object']]] for e in graph['edges']),
            distinct=sorted(sorted([index[a],index[b]]) for a,b in graph.get('distinct',[]))),sort_keys=True,separators=(',',':')))
    return min(versions)


def units(parsed):
    """Connected edges share one assignment; unresolved semantic units remain explicit."""
    edges=list({(e['subject'],e['predicate'],e['object']):e for e in parsed['edges']}.values())
    entities=parsed['entities'];adj={k:set() for k in entities}
    for e in edges:adj[e['subject']].add(e['object']);adj[e['object']].add(e['subject'])
    for a,b in parsed.get('distinct',[]):adj[a].add(b);adj[b].add(a)
    used={e[k] for e in edges for k in ('subject','object')};result={}
    while used:
        stack=[min(used)];component=set()
        while stack:
            name=stack.pop()
            if name in component:continue
            component.add(name);stack.extend(adj[name]-component)
        used-=component
        current=[e for e in edges if e['subject'] in component]
        g=dict(entities={n:entities[n] for n in component},edges=current,distinct=[p for p in parsed.get('distinct',[]) if set(p)<=component])
        if len(current)>3 or len(component)>6:
            key='opaque:'+identity(dict(caption=parsed['text'],reason='oversized_component'))
            result[key]=dict(key=key,opaque=True,weight=3,reason='oversized_component');continue
        key='graph:'+canonical(g)
        result[key]=dict(key=key,graph=g,weight=min(3,len(current)),opaque=False)
    for opaque in parsed.get('opaque',[]):
        key='opaque:'+identity(opaque['surface'])
        result[key]=dict(key=key,opaque=True,weight=min(3,max(1,opaque.get('weight',1))),reason=opaque['reason'],surface=opaque['surface'])
    return list(result.values())


def evaluate(unit,scene,bag=False):
    w=unit['weight']
    def answer(status,reason,witness=None):
        lo,hi=(w,w) if status=='supported' else (-w,-w) if status=='refuted' else (-w,w)
        return dict(status=status,lower=lo,upper=hi,reason=reason,witness=witness)
    if unit['opaque']:return answer('unknown','parser_opaque')
    graph=unit['graph'];nodes={n[0]:n[1] for n in scene['instances']};positive={tuple(e) for e in scene['positive_edges']}
    labels={(a,p):label for a,p,label in scene['actor_labels']}
    if bag:
        # Preserve identical partial-label rules; remove ONLY cross-edge identity
        # and inequality constraints, otherwise the contrast confounds binding
        # with a different treatment of explicit negative labels.
        statuses=[];capped=False
        for e in graph['edges']:
            g=dict(entities={n:graph['entities'][n] for n in (e['subject'],e['object'])},edges=[e],distinct=[])
            edge_result=evaluate(dict(opaque=False,weight=1,graph=g),scene)
            statuses.append(edge_result['status']);capped|=edge_result['reason']=='binding_cap'
        if all(s=='supported' for s in statuses):
            return answer('supported','bag_class_edges_ignore_joint_identity')
        if any(s=='refuted' for s in statuses):return answer('refuted','bag_independent_edge_explicit_zero')
        return answer('unknown','bag_binding_cap' if capped else 'bag_independent_edge_unknown')
    names=sorted(graph['entities']);domains=[[i for i,c in nodes.items() if c==graph['entities'][n]['category']] for n in names]
    if any(not domain for domain in domains):return answer('unknown','missing_class_not_absence')
    unknown=False;visited=0;possible=False;contradictions=[]
    for combination in itertools.product(*domains):
        assignment=dict(zip(names,combination))
        if any(assignment[a]==assignment[b] for a,b in graph.get('distinct',[])):continue
        possible=True;visited+=1
        if visited>POLICY['max_bindings']:return answer('unknown','binding_cap')
        states=[]
        for e in graph['edges']:
            s,o=assignment[e['subject']],assignment[e['object']];predicate=e['predicate']
            state=1 if (s,predicate,o) in positive else -1 if labels.get((s,predicate))==0 else 0
            states.append(state)
        if all(v==1 for v in states):return answer('supported','one_consistent_known_positive_assignment',assignment)
        if -1 not in states:unknown=True
        elif len(contradictions)<3:contradictions.append(assignment)
    # Missing/unannotated actors may add a compatible assignment. Never use link absence as false.
    if not possible or unknown or not scene['actor_inventory_complete']:
        return answer('unknown','unresolved_or_incomplete_actor_coverage')
    return answer('refuted','every_compatible_assignment_has_explicit_action_zero',dict(assignments_checked=visited,examples=contradictions))


def advantage(lower,upper):
    if not all(math.isfinite(x) for x in (lower,upper)) or lower>upper:raise ValueError('Invalid utility interval.')
    value=lower if lower>0 else upper if upper<0 else 0.
    return max(-1.,min(1.,value/3.))


def compare(sample,greedy,scene):
    su={u['key']:u for u in units(sample)};gu={u['key']:u for u in units(greedy)}
    added=sorted(su.keys()-gu.keys());removed=sorted(gu.keys()-su.keys());shared=sorted(su.keys()&gu.keys())
    evaluated={key:dict((su|gu)[key],**evaluate((su|gu)[key],scene)) for key in su.keys()|gu.keys()}
    lower=sum(evaluated[k]['lower'] for k in added)-sum(evaluated[k]['upper'] for k in removed)
    upper=sum(evaluated[k]['upper'] for k in added)-sum(evaluated[k]['lower'] for k in removed)
    zero=lambda r:0 if r['status']=='unknown' else r['lower']
    naive=sum(zero(evaluated[k]) for k in added)-sum(zero(evaluated[k]) for k in removed)
    bag={k:evaluate((su|gu)[k],scene,bag=True) for k in evaluated}
    bl=sum(bag[k]['lower'] for k in added)-sum(bag[k]['upper'] for k in removed)
    bu=sum(bag[k]['upper'] for k in added)-sum(bag[k]['lower'] for k in removed)
    a=advantage(lower,upper);az=advantage(naive,naive);ab=advantage(bl,bu)
    capped=any('binding_cap' in evaluated[k]['reason'] or 'binding_cap' in bag[k]['reason'] for k in added+removed)
    return dict(lower=lower,upper=upper,role_advantage=a,unknown_zero_advantage=az,bag_advantage=ab,
        search_capped=capped,raw_interval_difference=a!=az,raw_binding_difference=a!=ab,
        interval_operates=a!=az and not capped,binding_operates=a!=ab and not capped,operative=(a!=az or a!=ab) and not capped,
        added=added,removed=removed,shared=shared,units=evaluated)


def select_review(cases):
    chosen=[]
    for group in ([c for c in cases if c['comparison']['role_advantage']!=0],
                  [c for c in cases if c['comparison']['operative']],cases):
        existing={c['coco_id'] for c in chosen}
        chosen.extend(sorted((c for c in group if c['coco_id'] not in existing),key=lambda c:order(c['coco_id']))[:10])
    existing={c['coco_id'] for c in chosen}
    chosen.extend(sorted((c for c in cases if c['coco_id'] not in existing),key=lambda c:order(c['coco_id']))[:30-len(chosen)])
    return chosen
