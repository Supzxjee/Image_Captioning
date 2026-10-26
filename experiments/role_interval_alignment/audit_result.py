"""Audit cached review texts and hashes. No archive reread or model execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from role_logic import POLICY,identity,order,compare,select_review,advantage
from role_parser import PARSER_POLICY,translate

HERE=Path(__file__).resolve().parent


def main():
    cached=json.loads((HERE/'review/loaded_review.json').read_text(encoding='utf-8'))
    files,texts=cached['files'],cached['texts'];load=lambda name:json.loads(texts[name])
    cfg=load('protocol.json');receipt=load('receipt.json');cases=load('cases.json');summary=load('summary.json');inventory=load('inventory.json')
    package=json.loads((HERE/'PACKAGE.json').read_text(encoding='utf-8'))
    assert receipt['status']=='FEASIBILITY_PACKET_COMPLETE' and receipt['summary']==summary
    assert cfg['policy']==receipt['policy']==POLICY and cfg['parser_policy']==PARSER_POLICY==receipt['parser_policy']
    for name,expected in receipt['payload_sha256'].items():assert files[name]['sha256']==expected,name
    for name,expected in cfg['implementation_sha256'].items():
        assert files['source/'+name]['sha256']==expected==package['source_sha256'][name],name
    metadata=load('source/role_metadata.json');fit={r['coco_id']:r for r in metadata['fit']}
    assert cfg['metadata_sha256']==files['source/role_metadata.json']['sha256']
    selected=cfg['selected'];ids=[r['coco_id'] for r in selected]
    assert len(ids)==len(set(ids))==128
    assert ids==sorted(fit,key=order)[:128] and identity(ids)==cfg['selection_sha256']==receipt['selection_sha256']
    assert all(r==fit[r['coco_id']] for r in selected)
    reserved=set(sum(metadata['reserved'].values(),[]))
    assert not (set(cfg['planned_fit_ids'])&set(cfg['planned_replay_ids']))
    assert not (set(cfg['planned_fit_ids']+cfg['planned_replay_ids'])&reserved)
    assert len(cfg['planned_fit_ids'])==len(cfg['planned_replay_ids'])==inventory['planned_n']==1896
    records={}
    for rank in (0,1):
        p=load(f'complete_rank{rank}.json')
        for key,value in dict(rank=rank,protocol_sha256=identity(cfg),selection_sha256=cfg['selection_sha256'],
            stage_sha256=cfg['stage_sha256'],implementation_sha256=cfg['implementation_sha256'],records_sha256=identity(p['records'])).items():assert p[key]==value,key
        assert not p['optimizer_used'] and p['base_sha256']=='d2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e'
        assert [r['coco_id'] for r in p['records']]==ids[rank::2]
        for r in p['records']:
            assert r['seed']==42+r['coco_id'];records[r['coco_id']]=r
    assert len(cases)==len(records)==128 and {r['coco_id'] for r in cases}==set(ids)
    for r in cases:
        assert r['scene']==fit[r['coco_id']]
        for key in ('greedy','sample'):
            assert r[key]==records[r['coco_id']][key]
            assert identity(translate(r[key],r['parsed'][key]['dependency_tokens']))==identity(r['parsed'][key]['graph'])
        assert identity(compare(r['parsed']['sample']['graph'],r['parsed']['greedy']['graph'],r['scene']))==identity(r['comparison'])
    active=[r for r in cases if r['comparison']['role_advantage']!=0]
    operative=[r for r in cases if r['comparison']['operative']]
    assert len(active)==summary['active_pairs']==6 and len(operative)==summary['operative_pairs']==8
    assert sum(r['comparison']['binding_operates'] for r in cases)==summary['binding_operates']==0
    assert sum(r['comparison']['search_capped'] for r in cases)==0
    assert summary['status']=='FEASIBILITY_NO_GO' and len(active)<POLICY['min_active_pairs']
    manifest=load('review/manifest.json')
    assert identity(manifest['cases'])==identity(select_review(cases)) and len(manifest['cases'])==30
    parse_counts=Counter();reasons=Counter();status=Counter();refuted=[]
    for r in cases:
        for p in r['parsed'].values():
            g=p['graph'];parse_counts['opaque' if g['opaque'] else 'graph' if g['edges'] else 'out_of_scope']+=1
            reasons.update(o['reason'] for o in g['opaque'])
        for u in r['comparison']['units'].values():
            status[u['status']]+=1
            if u['status']=='refuted':
                categories={v['category'] for v in u['graph']['entities'].values()}
                assert categories&{'skateboard','snowboard','skis'}
                refuted.append(dict(coco_id=r['coco_id'],greedy=r['greedy'],sample=r['sample'],projected_categories=sorted(categories)))
    positive=[r['coco_id'] for r in active if r['comparison']['role_advantage']>0]
    negative=[r['coco_id'] for r in active if r['comparison']['role_advantage']<0]
    official=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/vcoco_train.json')
    raw=official.read_bytes();assert hashlib.sha256(raw).hexdigest()==metadata['sources']['vcoco_train']
    actions={a['action_name']:a for a in json.loads(raw)};ontology_checks=[]
    bycase={r['coco_id']:r for r in cases}
    for r in refuted:
        scene=bycase[r['coco_id']]['scene'];category=next(c for c in r['projected_categories'] if c!='person')
        action={'skateboard':'skateboard','snowboard':'snowboard','skis':'ski'}[category]
        actors=[n[0] for n in scene['instances'] if n[1]=='person'];assert len(actors)==1
        actor=actors[0];row=actions[action];j=row['ann_id'].index(actor);n=len(row['ann_id'])
        ride=actions['ride'];ride_j=ride['ann_id'].index(actor)
        assert row['image_id'][j]==r['coco_id'] and row['label'][j]==1 and ride['label'][ride_j]==0
        assert row['role_object_id'][j]==actor and len(row['role_object_id'])==n*len(row['role_name'])
        target=row['role_object_id'][j+n]
        assert any(aid==target and c==category for aid,c,_ in scene['instances'])
        ontology_checks.append(dict(coco_id=r['coco_id'],actor_id=actor,object_id=target,object_class=category,official_action=action,official_label=1,official_ride_label=0))
    corrected=[]
    for r in cases:
        v=r['comparison'];bounds={}
        for key,u in v['units'].items():
            bounds[key]=(-u['weight'],u['weight']) if u['status']=='refuted' else (u['lower'],u['upper'])
        lower=sum(bounds[k][0] for k in v['added'])-sum(bounds[k][1] for k in v['removed'])
        upper=sum(bounds[k][1] for k in v['added'])-sum(bounds[k][0] for k in v['removed'])
        a=advantage(lower,upper)
        if a:corrected.append(dict(coco_id=r['coco_id'],role_advantage=a,lower=lower,upper=upper))
    assert len(corrected)==2 and all(r['role_advantage']<0 for r in corrected)
    result=dict(zip_sha256=cached['zip_sha256'],technical_audit='PASS: payload/source/protocol/rank/selection/parser-translation/interval replay',
        receipt_payload_hashes_checked=len(receipt['payload_sha256']),runtime_sources_checked=len(cfg['implementation_sha256']),images=128,
        active_pairs=len(active),active_required=16,active_fraction=len(active)/128,operative_pairs=len(operative),operative_required=8,
        interval_operates=8,binding_operates=0,search_capped_pairs=0,parse_counts=dict(parse_counts),opaque_reasons=dict(reasons),unit_statuses=dict(status),
        explicit_graph_edge_counts=dict(Counter(len(p['graph']['edges']) for r in cases for p in r['parsed'].values() if p['graph']['edges'])),
        multi_person_scenes=sum(sum(n[1]=='person' for n in r['scene']['instances'])>1 for r in cases),
        official_vcoco_train_sha256=metadata['sources']['vcoco_train'],ontology_counterexamples=ontology_checks,
        posthoc_bookkeeping_only=dict(change='Replace the eight unlicensed refuted units with unknown; same stored captions/parse/selection; no new inference or gate validation.',active_pairs=len(corrected),positive_pairs=0,negative_pairs=2,cases=corrected),
        positive_advantage_ids=positive,negative_advantage_ids=negative,all_refuted_units=refuted,
        decision='NO_GO_CURRENT_RECIPE',training=False,human_audit_needed_to_decide_current_gate=False,
        explanation='Fails prespecified active density. Additionally all 8 refuted role units concern ride+skateboard/snowboard/skis; official ride=0 is not automatically English riding=false. Do not use these as caption error labels.',
        source_images_human_checked=False,caption_quality_metrics_computed=False,raw_stage_and_checkpoint_directly_verified=False,
        seconds=summary['elapsed_preparation_generation_scoring_seconds'])
    (HERE/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('all_refuted_units','opaque_reasons')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
