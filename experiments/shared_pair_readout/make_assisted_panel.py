"""Embed the visually proposed labels, never treating them as human ratings."""
import hashlib
import json
from pathlib import Path
from make_panel import OUT,HERE,assemble

def build():
    public,_,_=assemble()
    raw=(HERE/'ASSISTANT_PANEL_REVIEW.json').read_bytes();draft=json.loads(raw)
    ids={u['id'] for c in public['cards'] for u in c['captions']}
    assert draft['panel_sha256']==public['panel_sha256']
    assert draft['reviewer_type']=='assistant_proposed_not_human'
    assert set(draft['annotations'])==ids and len(ids)==175
    assert all(a['confirmed_by_human'] is False for a in draft['annotations'].values())
    allowed={'q_claim':{'yes','no','uncertain'},'target_relation':{'yes','no','uncertain'},'relation_correctness':{'correct','wrong','uncertain','not_applicable'}}
    for a in draft['annotations'].values():
        assert all(a[k] in choices for k,choices in allowed.items())
        assert not (a['q_claim']=='yes' and a['target_relation']!='yes')
        assert not (a['target_relation']=='no' and a['relation_correctness']!='not_applicable')
        assert not (a['target_relation']=='yes' and a['relation_correctness']=='not_applicable')
        assert not (a['target_relation']=='uncertain' and a['relation_correctness']!='uncertain')
    draft['proposal_sha256']=hashlib.sha256(raw).hexdigest()
    template=(HERE/'panel_template.html').read_text(encoding='utf8')
    encode=lambda x:json.dumps(x,ensure_ascii=False).replace('<','\\u003c')
    page=template.replace('__PAYLOAD__',encode(public)).replace('<script id="draft" type="application/json">null</script>','<script id="draft" type="application/json">'+encode(draft)+'</script>')
    assert '__PAYLOAD__' not in page and page.count('id="draft"')==1
    for c in public['cards']:assert (OUT/c['image']).is_file()
    (OUT/'assistant_review.html').write_text(page,encoding='utf8')
    print('175 AI proposals; 0 human confirmations. Page:',OUT/'assistant_review.html')

if __name__=='__main__':build()
