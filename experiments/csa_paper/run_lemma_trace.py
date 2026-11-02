"""USER-RUN CPU: reproduce exact POS settings on the frozen small packet."""
import argparse
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

from lemma_trace import inspect_doc, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out, report):
    import spacy
    here = Path(__file__).parent
    packet = json.loads((here/'trace_cases.json').read_text(encoding='utf-8'))
    original = json.loads((here/'source_environment.json').read_text(encoding='utf-8'))
    versions = {n: importlib.metadata.version(n) for n in ('spacy', 'thinc', 'numpy', 'en-core-web-sm')}
    if versions != {n: original['versions'][n] for n in versions}:
        raise ValueError('POS software versions differ from Output126')
    spacy.require_cpu()
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    if nlp.meta['version'] != '3.8.0' or set(nlp.pipe_names) != {'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer'}:
        raise ValueError('POS model/pipeline differs')
    observed = {str(p.relative_to(nlp.path)): sha(p) for p in sorted(Path(nlp.path).rglob('*')) if p.is_file()}
    if observed != original['pos_model_files_sha256']:
        raise ValueError('Installed POS model data hashes differ from Output126')
    report.update(versions=versions, model_files_matched=len(observed), model_data_identity_verified=True)
    records = []
    cases = packet['cases']
    docs = nlp.pipe((c['caption'] for c in cases), batch_size=256, n_process=1)
    for number, (case, doc) in enumerate(zip(cases, docs), 1):
        record = dict(case, **inspect_doc(doc))
        records.append(record)
        print(f'Case {number}/{len(cases)} [{case["kind"]}]: {case["caption"]}', flush=True)
        for focus in record['focus']:
            print(' ', focus['surface'], '→', focus['pos'], '/', focus['tag'], '→', focus['lemma'],
                  '; expected screen:', focus['expected_lemma_for_verb'], ';', focus['status'], flush=True)
    if len(records) != len(cases) or not all(r['focus'] for r in records):
        raise ValueError('Missing case or focus token in trace')
    report.update(summarize(records), status='LEMMA_TRACE_COMPLETE')
    (out/'trace_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    parts = []
    for number, record in enumerate(records, 1):
        rows = []
        for f in record['focus']:
            rows.append('<tr>'+''.join('<td>'+html.escape(str(f[k]))+'</td>' for k in
                ('surface', 'pos', 'tag', 'lemma', 'expected_lemma_for_verb', 'status'))+'</tr>')
        parts.append('<section><h2>'+str(number)+' · '+html.escape(record['kind'])+'</h2><p>'+html.escape(record['caption'])+
            '</p><table><tr><th>Từ</th><th>POS</th><th>Tag</th><th>Lemma thực tế</th><th>Lemma kiểm tra nếu VERB</th><th>Đối chiếu</th></tr>'+''.join(rows)+'</table></section>')
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPU lemma trace</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 18px;color:#17324d}section{margin:22px 0;padding:16px;background:#f3f6fa;border-radius:10px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccd7e1}</style><h1>Token → POS → lemma</h1><p>Giữ nguyên đầu ra spaCy. Đây là mẫu kiểm tra ngôn ngữ có chủ đích, không phải tỷ lệ lỗi toàn dataset. Không đổi bank hoặc huấn luyện mô hình caption.</p>'+''.join(parts)+'</html>'
    (out/'trace_review.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = dict(status='INCOMPLETE', bank_modified=False, roi_modified=False,
                  caption_model_executed=False, gpu_used=False, dataset_error_rate_estimated=False)
    start = time.monotonic()
    try:
        run(out, report)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['seconds'] = time.monotonic()-start
        (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
