"""Static geometry and browser-export evidence; never runs a model."""
from pathlib import Path
from itertools import combinations
import json, re, subprocess, hashlib
from PIL import Image
import fitz
ROOT=Path(__file__).resolve().parent
SKILL=Path('/home/agent/.codex/plugins/cache/diagram-design/diagram-design/2.6.142')

def transit(a,b,n):
    x,y,w,h=n['x'],n['y'],n['w'],n['h']
    if a[0]==b[0]: return x+.1<a[0]<x+w-.1 and min(max(a[1],b[1]),y+h-.1)>max(min(a[1],b[1]),y+.1)
    return y+.1<a[1]<y+h-.1 and min(max(a[0],b[0]),x+w-.1)>max(min(a[0],b[0]),x+.1)
def crossing(a,b,c,d):
    if a[1]==b[1] and c[0]==d[0]:
        if min(a[0],b[0])<c[0]<max(a[0],b[0]) and min(c[1],d[1])<a[1]<max(c[1],d[1]):return (c[0],a[1])
    if a[0]==b[0] and c[1]==d[1]:return crossing(c,d,a,b)
def overlap(a,b,c,d):
    if a[0]==b[0]==c[0]==d[0]:return min(max(a[1],b[1]),max(c[1],d[1]))>max(min(a[1],b[1]),min(c[1],d[1]))
    if a[1]==b[1]==c[1]==d[1]:return min(max(a[0],b[0]),max(c[0],d[0]))>max(min(a[0],b[0]),min(c[0],d[0]))
    return False

reports={}; issues=[]
for fig in json.loads((ROOT/'diagram_geometry.json').read_text()):
    own=[];nodes=fig['nodes'];routes=fig['routes'];slug=fig['slug']
    # Master embeds already checked panels; its routes here are zoom guides only.
    for r in routes:
        for a,b in zip(r['points'],r['points'][1:]):
            if a[0]!=b[0] and a[1]!=b[1]:own.append(['diagonal',r['id']])
            for n in nodes:
                if n['id'] not in [r['src'],r['dst']] and transit(a,b,n):own.append(['transit',r['id'],n['id']])
    for ra,rb in combinations(routes,2):
        for a,b in zip(ra['points'],ra['points'][1:]):
            for c,d in zip(rb['points'],rb['points'][1:]):
                pt=crossing(a,b,c,d)
                if pt:own.append(['crossing',ra['id'],rb['id'],pt])
                if overlap(a,b,c,d):own.append(['overlap',ra['id'],rb['id']])
    for a,b in combinations(nodes,2):
        if min(a['x']+a['w'],b['x']+b['w'])>max(a['x'],b['x']) and min(a['y']+a['h'],b['y']+b['h'])>max(a['y'],b['y']):own.append(['node-overlap',a['id'],b['id']])
    reports[slug]=dict(nodes=len(nodes),routes=len(routes),issues=own)
    issues.extend([[slug]+v for v in own])

render_path=ROOT/'render_audit.json'
if render_path.exists():
    renders=json.loads(render_path.read_text())
    for slug,d in renders.items():
        if d.get('html_sha256')!=hashlib.sha256((ROOT/(slug+'.html')).read_bytes()).hexdigest():issues.append([slug,'stale-render'])
        for field in ['overflow','iconTextOverlap','looseTextStrokeOverlap','outsideCanvas']:
            if d[field]:issues.append([slug,field,d[field]])
    for fig in json.loads((ROOT/'diagram_geometry.json').read_text()):
        slug=fig['slug']
        if slug not in renders:issues.append([slug,'missing-render'])
        for suffix in ['.svg','.png','.pdf']:
            if not (ROOT/(slug+suffix)).exists():issues.append([slug,'missing-export',suffix])
        if (ROOT/(slug+'.png')).exists():
            with Image.open(ROOT/(slug+'.png')) as im:
                if im.size!=(fig['w']*2,fig['h']*2):issues.append([slug,'png-size',im.size])
else:issues.append(['missing-render-audit'])

pdfs={}
for f in ROOT.glob('*.pdf'):
    with fitz.open(f) as doc:
        pdfs[f.name]=dict(pages=len(doc),selectable_text=sum(len(p.get_text()) for p in doc),vector_drawings=sum(len(p.get_drawings()) for p in doc),raster_images=sum(len(p.get_images()) for p in doc))
        expected_pages=8 if f.name=='captioning-mechanisms-v4-panels.pdf' else 1
        if len(doc)!=expected_pages or pdfs[f.name]['selectable_text']<100 or pdfs[f.name]['vector_drawings']<10:issues.append(['pdf',f.name])

content=(ROOT/'captioning-mechanisms-v4.html').read_text()
checks={
    'two_slot_mlps':content.count('>Slot MLP<')==2,
    'no_forbidden_method_names':not re.search(r'\b(?:CSA|CSA-Net|OSA|ORA|BLIP|RoIAlign)\b',content),
    'concat_and_original_region_inputs':all(v in content for v in ['[tₒ; Vₒ[i]; bᵢ]','[tᵣ; Vₒ[s]; Vₒ[o]; bₛ; bₒ]','Both slot builders use original Vₒ']),
    'three_fusions':all(v in content for v in ['Object fusion 1','Object fusion 2','Relation fusion']),
    'adapter_empty_and_zero_init':content.count('Zero-initialized')==2 and 'No valid linked slot → zero residual' in content,
    'decoder_raw_e_and_separate_memories':all(v in content for v in ['Gates use raw word embedding e, without PE.','Q = H · K = V = Vₒ′','Q = H · K = V = Vᵣ′']),
    'rerank_and_training':all(v in content for v in ['0.3 D(y) + 0.4 C(y) + 0.3 O(y)','No length','normalization.','Equal scores: earlier beam wins.','Total loss = LCE + 0.05 LSCST']),
    'one_clean_photo':content.count('<image ')==1,
    'stage2_current_batch_XE_and_policy_logp':all(v in content for v in ['Current-batch LCE','TRAIN teacher forcing','Rollout log p(yₛ)']),
}
for k,v in checks.items():
    if not v:issues.append(['content',k])
for f in ROOT.glob('*.html'):
    r=subprocess.run(['python',str(SKILL/'skills/diagram-design/scripts/self_check.py'),str(f)],capture_output=True,text=True)
    if r.returncode:issues.append(['skill-self-check',f.name,r.stdout,r.stderr])
result=dict(status='PASS' if not issues else 'FAIL',figures=reports,content_checks=checks,pdfs=pdfs,issues=issues,limits='Static and graphics only. No model or notebook was run. Master checks its zoom guides; translated panels and overview are checked separately. Named tensor ports avoid long crossing buses.')
(ROOT/'figure_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],issues=issues),ensure_ascii=False,indent=2))
raise SystemExit(bool(issues))
