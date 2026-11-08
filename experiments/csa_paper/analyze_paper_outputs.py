"""Read existing text caches only: costs and deterministic CHAIR review shortlist."""
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2] / '.review_cache'
OUT = HERE / 'paper_analysis'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    manifest = read(HERE/'PROMPT_TEST_EXTRA_INPUTS.json')
    scst = read(HERE/'PROMPT_SCST_RESULT.json')
    xe = read(HERE/'PROMPT_SCRATCH_XE10_RESULT.json')
    files, costs = {}, {}
    for arm in ('CONTROL', 'PROMPT'):
        saved = read(CACHE/'prompt_TEST'/(arm+'.json'))
        files[arm] = saved['files']
        f = files[arm]
        for name, digest in manifest['arms'][arm]['payload_sha256'].items():
            assert hashlib.sha256(f[name].encode('utf-8')).hexdigest() == digest, name
        scratch = read(CACHE/'prompt_scratch_xe10'/(arm+'.json'))['files']
        runtime = json.loads(scratch['runtime.json'])
        assert runtime['arm'] == arm and runtime['world_size'] == 2
        timings = {}
        for phase, prefix in [('beam5', 'normal_decode_rank'), ('rerank', 'rerank_worker')]:
            ranks = [json.loads(f[prefix+str(rank)+'.json']) for rank in (0, 1)]
            assert all(r['complete'] and r['images'] == 2500 and r['seconds'] > 0 for r in ranks)
            timings[phase] = dict(rank_seconds=[r['seconds'] for r in ranks],
                                  max_rank_seconds=max(r['seconds'] for r in ranks))
        costs[arm] = dict(trainable_caption_parameters=runtime['trainable_parameters'],
            xe10_worker_seconds=xe['arms'][arm]['worker_seconds'],
            scst_full1_worker_seconds=scst['arms'][arm]['worker_seconds'],
            test5000_phases=timings)
    costs['parameter_delta'] = costs['PROMPT']['trainable_caption_parameters']-costs['CONTROL']['trainable_caption_parameters']
    costs['limits'] = ['Frozen CLIP/YOLO/UPT and rerank encoder parameters excluded from caption trainable count.',
        'Two T4 workers: max rank seconds is a phase proxy, not measured end-to-end latency.',
        'Preprocessing, prompt/feature extraction, model loading, barriers and CPU scoring excluded from inference timing.',
        'Worker times from separate sessions; not a controlled speed benchmark or GPU quota consumption.',
        'XE workers ran 10 epochs; selected SCST starts C10/P9.']
    write(OUT/'costs.json', costs)
    notes = ['# Existing output cost summary', '', '| Arm | Trainable caption params | XE10 worker hours | SCST full1 worker hours | Beam5 max-rank s | Rerank max-rank s |', '|---|---:|---:|---:|---:|---:|']
    for arm in ('CONTROL', 'PROMPT'):
        c=costs[arm]
        notes.append(f"| {arm} | {c['trainable_caption_parameters']:,} | {c['xe10_worker_seconds']/3600:.3f} | {c['scst_full1_worker_seconds']/3600:.3f} | {c['test5000_phases']['beam5']['max_rank_seconds']:.3f} | {c['test5000_phases']['rerank']['max_rank_seconds']:.3f} |")
    notes += ['', *['- '+s for s in costs['limits']], '', 'Sources: pinned TEST184/61 text caches; XE178/58 runtime; SCST180/60 audited metadata. No model or official scorer executed.']
    (OUT/'COSTS.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    items, counts = [], {}
    ref = {r['coco_id']:r['references'] for r in json.loads(files['CONTROL']['test_references.json'])}
    inp = {r['coco_id']:r for r in json.loads(files['CONTROL']['inference_inputs.json'])}
    for policy in ('normal', 'fixed_rerank'):
        chair = {a:{r['image_id']:r for r in json.loads(files[a][policy+'_chair_output.json'])['sentences']} for a in files}
        assert set(chair['CONTROL']) == set(chair['PROMPT']) == set(ref) and len(ref)==5000
        groups = {'flag_removed':[], 'flag_introduced':[], 'neither_flagged':[], 'both_flagged':[]}
        for i in sorted(ref):
            c,p = [bool(chair[a][i]['mscoco_hallucinated_words']) for a in ('CONTROL', 'PROMPT')]
            group = 'flag_removed' if c and not p else 'flag_introduced' if p and not c else 'both_flagged' if c else 'neither_flagged'
            groups[group].append(i)
        counts[policy] = {g:len(ids) for g,ids in groups.items()}
        for group, ids in groups.items():
            selected = sorted(ids, key=lambda i:hashlib.sha256(f'paper-review-v1/{policy}/{i}'.encode()).hexdigest())[:3]
            for i in selected:
                filename=Path(inp[i]['image_path']).name
                subset='train2014' if 'train2014' in filename else 'val2014'
                items.append(dict(coco_id=i, policy=policy, group=group,
                    image_url=f'https://images.cocodataset.org/{subset}/{filename}',
                    CONTROL={k:chair['CONTROL'][i][k] for k in ('caption','mscoco_hallucinated_words')},
                    PROMPT={k:chair['PROMPT'][i][k] for k in ('caption','mscoco_hallucinated_words')},
                    references=ref[i], visually_verified=False))
    write(OUT/'example_shortlist.json', dict(status='AUTOMATED_SHORTLIST_NOT_VISUAL_TRUTH', group_counts=counts,
        selection='3 per CHAIR flag transition per policy, deterministic SHA order; not best examples or representative sample.', items=items))
    cards=[]
    for row in items:
        esc=lambda v:html.escape(str(v))
        cards.append(f"<article><h2>{row['coco_id']} — {row['policy']} — {row['group']}</h2><img loading='lazy' src='{row['image_url']}' alt='COCO image; requires Internet'><p><b>CONTROL:</b> {esc(row['CONTROL']['caption'])}</p><p><b>PROMPT:</b> {esc(row['PROMPT']['caption'])}</p><p>CHAIR words C: {esc(row['CONTROL']['mscoco_hallucinated_words'])}; P: {esc(row['PROMPT']['mscoco_hallucinated_words'])}</p><details><summary>References</summary>{'<br>'.join(esc(s) for s in row['references'])}</details><p>Visual confirmation: pending. CHAIR does not assess every relation/attribute.</p></article>")
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Caption comparison shortlist</title><style>body{font:17px Arial;max-width:1050px;margin:35px auto;background:#f4f6f8}article{background:white;padding:24px;margin:20px 0}img{max-width:500px;max-height:350px}h2{font-size:20px}</style><h1>Automated shortlist for visual review</h1><p>24 cases: 3 per transition × 4 transitions × 2 policies. Not manually verified; no claim that unflagged captions are correct. Images load from COCO with Internet.</p>'+''.join(cards),encoding='utf-8')
    print('Stored data checks PASS; costs and 24 unverified examples saved.', counts)


if __name__ == '__main__': main()
