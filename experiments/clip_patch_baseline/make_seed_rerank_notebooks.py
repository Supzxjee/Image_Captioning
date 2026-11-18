"""Build fixed-rerank notebooks from the successful TEST and seed42 notebooks."""
import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def code(nb, index):
    return ''.join(nb['cells'][index]['source'])


def embedded_node(tree, filename):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == 'write_text'
               and isinstance(n.func.value, ast.BinOp)
               and isinstance(n.func.value.right, ast.Constant)
               and n.func.value.right.value == filename]
    if len(matches) != 1:
        raise ValueError('Expected one embedded source: ' + filename)
    return matches[0]


def embedded(source, filename):
    return ast.literal_eval(embedded_node(ast.parse(source), filename).args[0])


def function_source(source, name):
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def build(seed, historical):
    original = json.loads((HERE / f'CSA_PER_CAPTION_SEED{seed}_TEST5000_API_2GPU.ipynb').read_text(encoding='utf-8'))
    checked = json.loads((HERE / f'CSA_PER_CAPTION_SEED{seed}_TEST5000_RESULT.json').read_text(encoding='utf-8'))
    nb = copy.deepcopy(original)
    output_name = f'csa_per_caption_seed{seed}_rerank_test5000'
    checkpoint_source = checked['result']['source_version']
    test_source = checked['test_output_source']
    nb['cells'][0]['source'] = [
        f'# Seed {seed}: fixed rerank on Karpathy TEST5000\n\n'
        f'Checkpoint {checkpoint_source}; original TEST {test_source}. '
        'The original output saved top1 only. Regenerate five beams and require exact prior top1 SHA replay before CLIP scoring. '
        'No training or weight changes. Fixed decoder/CLIP/object weights 0.3/0.4/0.3; no TEST tuning.\n\n'
        'Select GPU T4 x2 and Internet. Attach visual-cache, mscoco-2014, csa-prompt-test-cache-v181 '
        'and objectdetectionecache. Enable Kaggle Secret codex or KAGGLE_API_TOKEN with access to the checkpoint source. '
        'Do not attach a checkpoint output; the default downloads only exact best.bin through API. '
        f'Save & Run All and send {output_name}_review.zip. '
        'Preserve the full output, including beams and CLIP scores. Three-hour STOP cap, not an ETA. '
        'Both top1 and rerank receive all eight metrics.\n']
    setup = code(original, 1).replace('WALL_HOURS=2', 'WALL_HOURS=3')
    setup = setup.replace(f"csa_per_caption_seed{seed}_test5000'", output_name + "'")
    setup += f'\nTEST_OUTPUT_VERSION={test_source!r}\n'
    setup += "DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')\n"
    setup += "if not DETECTIONS.is_file():raise FileNotFoundError('Attach objectdetectionecache or correct DETECTIONS')\n"
    setup += "os.environ.update(TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',HF_HUB_DISABLE_XET='1')\n"
    nb['cells'][1]['source'] = [setup]

    old = code(historical, 3)
    source = code(original, 3)
    prefix = source[:source.index("    run([sys.executable,'-u',SOURCE/'run_csa_per_caption_test.py','tokenize'")]
    tail = old[old.index("    for path,digest in {'captioning/object_consistency.py'"):old.index('except BaseException:')]
    suffix = source[source.index('except BaseException:'):]
    tree = ast.parse(prefix + tail + suffix)
    runner = embedded_node(tree, 'run_csa_per_caption_test.py')
    runner_source = ast.literal_eval(runner.args[0])
    prior_worker = function_source(runner_source, 'worker')
    beam_worker = function_source(embedded(old, 'run_csa_per_caption_test.py'), 'worker')
    runner.args[0] = ast.Constant(runner_source.replace(prior_worker, beam_worker))

    replay = dict(checkpoint_sha256=checked['result']['checkpoint_sha256'],
                  rank_caption_sha256=[checked[f'worker{i}']['caption_sha256'] for i in (0, 1)],
                  reference_sha256=checked['scoring_lock']['files_sha256']['test_references.json'],
                  test_output_source=test_source)
    embedded_node(tree, 'REPLAY_RECEIPTS.json').args[0] = ast.parse(f'json.dumps({replay!r})', mode='eval').body
    helper = embedded_node(tree, 'csa_per_caption_rerank_data.py')
    helper_source = ast.literal_eval(helper.args[0])
    helper_source = helper_source.replace(
        "    if source['checkpoint_sha256'] != replay['checkpoint_sha256']:",
        "    if sha(test/'test_references.json') != replay['reference_sha256']:\n"
        "        raise ValueError('Changed original TEST reference bytes')\n"
        "    if source['checkpoint_sha256'] != replay['checkpoint_sha256']:", 1)
    helper.args[0] = ast.Constant(helper_source)
    source = ast.unparse(tree) + '\n'
    marker = "    complete = result['status'] == 'CSA_PER_CAPTION_FIXED_RERANK_TEST5000_COMPLETE'"
    if source.count(marker) != 1:
        raise ValueError('Missing rerank completion guard')
    replay_check = f'''    expected_top1 = {checked['result']['metrics']!r}
    for key, value in expected_top1.items():
        if key != 'NLL' and abs(result['metrics']['decoder_rank0'][key] - value) > 1e-7:
            raise ValueError('Original TEST metric replay differs: ' + key)
    workers = [json.loads((OUTPUT/f'worker{{rank}}.json').read_text()) for rank in (0, 1)]
    nll = sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
    for policy_metrics in result['metrics'].values():
        policy_metrics['NLL'] = nll
    result.update(seed=SEED, checkpoint_source=SOURCE_VERSION, caption_test_output=TEST_OUTPUT_VERSION,
                  verified_original_top1_receipts=True, verified_original_top1_metrics=True,
                  NLL_note='Same frozen captioner NLL for both output selection policies.')
    (rerank_out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
'''
    source = source.replace(marker, replay_check + marker)
    source = source.replace('source_version=SOURCE_VERSION)', 'source_version=SOURCE_VERSION, caption_test_output=TEST_OUTPUT_VERSION, reranking=True)')
    nb['cells'][3]['source'] = [source]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'seed{seed}-cell', 'exec')
            cell['execution_count'] = None
            cell['outputs'] = []
    for filename in ('csa_per_caption_test_data.py', 'run_csa_per_caption_test.py',
                     'csa_per_caption_rerank_data.py', 'run_csa_per_caption_rerank.py'):
        compile(embedded(source, filename), filename, 'exec')
    return nb


if __name__ == '__main__':
    historical = json.loads((HERE / 'CLIP_PATCH_CSA_PER_CAPTION_BEST9_RERANK_TEST5000_2GPU.ipynb').read_text(encoding='utf-8'))
    for seed in (43, 44):
        notebook = build(seed, historical)
        path = HERE / f'CSA_PER_CAPTION_SEED{seed}_RERANK_TEST5000_API_2GPU.ipynb'
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(path.name)
