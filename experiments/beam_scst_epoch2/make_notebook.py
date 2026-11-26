"""Package the audited epoch1 recipe, explicit epoch2 continuation and TEST."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'beam_scst_epoch1'
spec = importlib.util.spec_from_file_location('epoch1_packaging', OLD/'make_notebook.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
literals, once = previous.literals, previous.once

NAME = 'BEAM_SCST_EPOCH2_FROM_V199_VALID_TEST_RERANK_2GPU.ipynb'


def notebook(name):
    return json.loads((OLD/name).read_text(encoding='utf-8'))


def replace_function(source, name, body):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+body+'\n'+''.join(lines[node.end_lineno:])


def worker(payload):
    s = payload['run_beam_epoch.py']
    s = replace_function(s, 'validate_resume', '')
    s = once(s, 'from full_epoch_data import FullTrain', '''from full_epoch_data import FullTrain
from speed_candidate import batch_rows, train_beams_fast
from epoch2_contract import (PARENT_SHA, PARENT_VERSION, PARENT_STEPS, END_STEPS,
    validate_parent, make_identity, validate_resume)''')
    start = s.index('        configure_valid(cfg)')
    end = s.index('        class Policy(nn.Module):', start)
    s = s[:start]+'''        configure_valid(cfg)
        source = Path(cfg['checkpoint_root'])
        if sha(source) != PARENT_SHA:
            raise ValueError('Need exact v199 full checkpoint including Adam/RNG')
        ck = torch.load(source, map_location='cpu', weights_only=False)
        expected = read(Path(__file__).with_name('PARENT_CHECKPOINT_RECEIPT.json'))
        validate_parent(ck, expected)
        words = read(out/'vocabulary.json')
        if len(words) != 10000:raise ValueError('Wrong vocabulary')
        identity = make_identity(cfg, expected, str(torch.__version__), np.__version__, sha(out/'vocabulary.json'))
        model = CSAPatchCaptioner(len(words), ck['model']['core.nouns'], ck['model']['core.verbs'])
        model.load_state_dict(ck['model'], strict=True)
        if sum(p.numel() for p in model.parameters()) != 53183736:
            raise ValueError('Captioner architecture differs')
        model = model.to(device).train()
        data = FullTrain(cfg['train_roots'])
        rewarder = load_reward(cfg['train_roots'][0])
        if set(map(int, np.concatenate(data.ids))) != set(rewarder.references):
            raise ValueError('Cached TRAIN reward and feature cohorts differ')

'''+s[end:]
    s = s.replace('return train_beams(self.captioner, self.captioner.encode(patches, prompts))',
                  'return train_beams_fast(self.captioner, self.captioner.encode(patches, prompts))')
    s = s.replace('seq, logps = train_beams(model, memory)', 'seq, logps = train_beams_fast(model, memory)')
    start = s.index('        cursor = updates = 0')
    end = s.index('        save(out/f\'identity_rank{rank}.json\', identity)', start)
    s = s[:start]+'''        cursor, updates = 0, PARENT_STEPS
        prior = ck
        resume_receipt = None
        if cfg.get('resume_root'):
            root = Path(cfg['resume_root'])
            resume_receipt = read(root/'checkpoint_receipt.json')
            validate_resume(resume_receipt, identity)
            if sha(root/'checkpoint.bin') != resume_receipt['sha256']:
                raise ValueError('Changed epoch2 resume binary')
            prior = torch.load(root/'checkpoint.bin', map_location='cpu', weights_only=False)
            for key in ('identity','updates','cursor_images','epoch','parent_sha256','source_sha256'):
                if prior[key] != resume_receipt[key]:raise ValueError('Resume binary/receipt differ: '+key)
            cursor, updates = prior['cursor_images'], prior['updates']
        model.load_state_dict(prior['model'], strict=True)
        optimizer.load_state_dict(prior['optimizer'])
        for group in optimizer.param_groups:
            if (group['lr'] != 4e-6 or tuple(group['betas']) != (.9,.99)
                    or group['eps'] != 1e-8 or group['weight_decay'] != 0):
                raise ValueError('Changed parent Adam settings')
        if not optimizer.state:raise ValueError('Missing Adam moments')
        steps = [float(state['step']) for state in optimizer.state.values()]
        if not all(math.isfinite(x) and 0 < x <= updates for x in steps):
            raise ValueError('Invalid Adam step counters')
        state = prior['rng'][rank]
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'], device)
        np.random.set_state(state['numpy'])
        if not torch.equal(torch.get_rng_state(),state['cpu']) or not torch.equal(torch.cuda.get_rng_state(device),state['cuda']):
            raise ValueError('RNG restore failed')
        save(out/f'continuity_rank{rank}.json',dict(parent_version=PARENT_VERSION,parent_sha256=PARENT_SHA,
            epoch=2,cursor_images=cursor,updates=updates,optimizer_loaded=True,optimizer_states=len(steps),
            Adam_step_min=min(steps),Adam_step_max=max(steps),rank_RNG_restored=True,
            preflight_only=mode=='preflight',resume=resume_receipt))
        del prior, ck
'''+s[end:]
    s = once(s, "                    identity=identity, rng=states), temp)",
             "                    identity=identity, epoch=2, parent_sha256=PARENT_SHA, rng=states), temp)")
    s = once(s, "                    complete_epoch=cursor == IMAGES, test_used=False))",
             "                    epoch=2, parent_sha256=PARENT_SHA, complete_epoch=cursor == IMAGES, test_used=False))")
    s = once(s, 'x, prompt, ids = data.batch(rows, device)', 'x, prompt, ids = batch_rows(data, rows, device)')
    s = once(s, '        while cursor < IMAGES:', '''        if mode == 'train':checkpoint_now()
        while cursor < IMAGES:''')
    s = once(s, '        if updates != STEPS:', '        if updates != END_STEPS:')
    s = once(s, "torch.save(dict(model=model.state_dict(), updates=updates, source_sha256=SOURCE_SHA), tmp)",
             "torch.save(dict(model=model.state_dict(), updates=updates, source_sha256=SOURCE_SHA, epoch=2, parent_sha256=PARENT_SHA), tmp)")
    s = once(s, '                source_sha256=SOURCE_SHA, cursor_images=cursor, identity=identity))',
             '                source_sha256=SOURCE_SHA, cursor_images=cursor, epoch=2, parent_sha256=PARENT_SHA, complete_epoch=True, identity=identity))')
    s = once(s, '                epochs=1, source_sha256=SOURCE_SHA, test_used=False, reranking=False,',
             '                epochs=2, epoch2_image_events=cursor, cumulative_image_events=2*IMAGES, parent_sha256=PARENT_SHA, source_sha256=SOURCE_SHA, test_used=False, reranking=False,')
    s = s.replace('One-epoch endpoint mismatch', 'Epoch2 endpoint must reach cumulative7082 updates')
    s = once(s, "        del prior, ck", "        del state, prior, ck")
    s = s.replace('from csa_per_caption_test_data import checkpoint, validate_checkpoint_state, configure_valid',
                  'from csa_per_caption_test_data import configure_valid')
    (HERE/'run_epoch2.py').write_text(s, encoding='utf-8')
    return s


def build():
    train_nb = notebook('BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb')
    test_nb = notebook('BEAM_SCST_V199_TEST5000_RERANK_API_2GPU.ipynb')
    train = literals(''.join(train_nb['cells'][1]['source']))['SOURCE_PAYLOADS'].copy()
    test_values = literals(''.join(test_nb['cells'][1]['source']))
    test = test_values['SOURCE_PAYLOADS'].copy()
    train['run_epoch2.py'] = worker(train)
    del train['run_beam_epoch.py']
    for name in ('speed_candidate.py','PARENT_CHECKPOINT_RECEIPT.json'):
        train[name] = (OLD/name).read_text(encoding='utf-8')
    for package in (train, test):
        package['epoch2_contract.py'] = (HERE/'epoch2_contract.py').read_text(encoding='utf-8')
        package['EPOCH2_PROTOCOL.md'] = (HERE/'PROTOCOL.md').read_text(encoding='utf-8')
    # Runtime binding is written only after a sealed, complete epoch2 endpoint exists.
    data = test['csa_per_caption_test_data.py']
    data = once(data, "CHECKPOINT_SHA = '3b03acc29f025b07dc358846b30618ae9c83f8634bc12bf7905e9d908b971f24'", '''from epoch2_contract import validate_eval_binding, END_STEPS, PARENT_SHA
EPOCH2_BINDING = validate_eval_binding(read(HERE/'EPOCH2_EVAL_LOCK.json'))
CHECKPOINT_SHA = EPOCH2_BINDING['sha256']''')
    data = once(data, "VERSION = 'ducanh3/notebook7bc83c4c76/199'", "VERSION = EPOCH2_BINDING['version']")
    data = data.replace("value['epoch'] != 1", "value['epoch'] != 2").replace("value['updates'] != 3541", "value['updates'] != END_STEPS")
    data = data.replace('epoch=1, updates=3541', 'epoch=2, updates=END_STEPS')
    data = once(data, "    if ck['updates'] != 3541 or ck['source_sha256'] != XE_SOURCE_SHA:",
                "    if ck['updates'] != END_STEPS or ck['source_sha256'] != XE_SOURCE_SHA or ck['epoch'] != 2 or ck['parent_sha256'] != PARENT_SHA:")
    data = data.replace('beam SCST199', 'beam SCST epoch2')
    test['csa_per_caption_test_data.py'] = data
    del test['REPLAY_RECEIPTS.json']
    hashes = lambda d: {n:hashlib.sha256(s.encode()).hexdigest() for n,s in d.items()}
    references = dict(CE=test_values['CE_REFERENCE'], epoch1_TEST=json.loads((OLD/'TEST_RESULT.json').read_text())['SCST_metrics'],
                      epoch1_VALID=json.loads((OLD/'RUN1_RESULT.json').read_text())['metrics'], selector=test_values['FROZEN_SELECTOR'])
    header = (HERE/'notebook_config.py').read_text(encoding='utf-8')
    header += '\nTRAIN_PAYLOADS='+repr(train)+'\nTEST_PAYLOADS='+repr(test)+'\n'
    header += 'TRAIN_SHA256='+repr(hashes(train))+'\nTEST_SHA256='+repr(hashes(test))+'\n'
    header += 'REFERENCES='+repr(references)+'\n'
    header += 'ROUTING_PREFLIGHT='+repr(literals(''.join(train_nb['cells'][1]['source']))['ROUTING_PREFLIGHT'])+'\n'
    nb = copy.deepcopy(train_nb)
    nb['cells'][0]['source'] = (HERE/'NOTEBOOK_GUIDE.md').read_text(encoding='utf-8').splitlines(keepends=True)
    nb['cells'][1]['source'] = header.splitlines(keepends=True)
    helpers = ''.join(train_nb['cells'][2]['source']).replace('TEST/CPU metrics active;', 'Epoch2 train/evaluation active;')
    nb['cells'][2]['source'] = helpers.splitlines(keepends=True)
    nb['cells'][3]['source'] = (HERE/'notebook_run.py').read_text(encoding='utf-8').splitlines(keepends=True)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell{i}', 'exec')
            cell['execution_count'] = None;cell['outputs'] = []
    for name,source in {**train,**test}.items():
        if name.endswith('.py'):compile(source,name,'exec')
    target = HERE/NAME
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    manifest = dict(notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    train_sha256=hashes(train),test_sha256=hashes(test),references=references,
                    parent_version='ducanh3/notebook7bc83c4c76/199',full_checkpoint_sha256=json.loads(train['PARENT_CHECKPOINT_RECEIPT.json'])['sha256'],
                    endpoint_updates=7082,epoch2_images=113287,wall_hours=5.75,user_GPU_hours=6,
                    API_or_model_executed=False)
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(NAME,len(train),'train files;',len(test),'TEST files; unrun')


if __name__ == '__main__':build()
