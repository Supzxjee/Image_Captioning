"""Human-readable free-generation review; no keyword-based semantic grading."""
import html


def render(manifest,phases,path):
    esc=lambda v:html.escape(str(v),quote=True)
    lookup={phase:{r['group_id']:r for r in records} for phase,records in phases.items()}
    held=set(manifest['partitions']['adaptation_holdout'])
    parts=['<!doctype html><meta charset="utf-8"><title>Phrase decoder pilot review</title><style>body{font:16px/1.5 Arial;background:#eef3f7;color:#13283c;max-width:1250px;margin:auto;padding:24px}section{background:white;border-radius:10px;padding:20px;margin:20px 0}.events{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pic{position:relative}.pic img{width:100%;display:block}.pic svg{position:absolute;inset:0;width:100%;height:100%}td,th{padding:8px;border:1px solid #ccc;text-align:left}table{border-collapse:collapse;width:100%}@media(max-width:800px){.events{grid-template-columns:1fr}}</style>',
        '<h1>Kiểm tra cụm sinh tự do</h1><p>Scoring đúng không có nghĩa cụm sinh đúng. Xem đúng cặp S xanh/O cam; ghi nhận paraphrase, sai instance, câu chung chung và khẳng định thêm sai. Không dùng khớp động từ để tự chấm.</p>']
    for group in sorted(manifest['groups'],key=lambda g:(g['group_id'] not in held,g['group_id'])):
        key=group['group_id'];split='ADAPTATION HOLDOUT (7 groups, không test độc lập)' if key in held else 'FIT (14 groups)'
        parts.append(f'<section><h2>{esc(key)}</h2><p>{esc(split)}</p><p>C1: {esc(group["descriptions"][0])}<br>C2: {esc(group["descriptions"][1])}</p><div class="events">')
        for i,e in enumerate(group['events']):
            rect=[]
            for b,color,label in zip((e['subject_box'],e['object_box']),('lime','orange'),('S','O')):
                rect.append(f'<rect x="{b[0]*100}" y="{b[1]*100}" width="{(b[2]-b[0])*100}" height="{(b[3]-b[1])*100}" fill="none" stroke="{color}" stroke-width=".6"/><text x="{b[0]*100+1}" y="{max(4,b[1]*100+4)}" fill="{color}" font-size="4">{label}</text>')
            parts.append(f'<div><h3>Event{i+1} · COCO {e["image_id"]}</h3><div class="pic"><img src="images/{esc(e["filename"])}"><svg viewBox="0 0 100 100" preserveAspectRatio="none">{"".join(rect)}</svg></div><p>Human target: C{group["positive_indices"][i]+1}</p></div>')
        parts.append('</div><table><tr><th>Thời điểm/điều kiện</th><th>Event1</th><th>Event2</th></tr>')
        for phase in ('before','after'):
            for condition in ('on','off'):
                record=lookup[phase][key]['conditions'][condition]
                parts.append(f'<tr><th>{phase}/{condition}</th>'+''.join(f'<td>{esc(text)}{" [TRUNCATED]" if record["truncated"][i] else ""}<br><small>Positive−negative mean logP: {record["positive_margins"][i]:.5f}</small></td>' for i,text in enumerate(record['captions']))+'</tr>')
        parts.append('</table></section>')
    path.write_text(''.join(parts),encoding='utf-8')
