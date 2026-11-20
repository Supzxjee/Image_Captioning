"""Run only the upstream PaperOrchestra diagram branch; no research models."""
import argparse
import base64
from datetime import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('C:/Users/leduc/.codex/tools/paper-orchestra')
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Call Gemini; without this only check inputs.')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    materials = HERE / 'materials'
    assert (materials / 'idea_sparse.md').is_file()
    ready = bool(os.getenv('GEMINI_API_KEY'))
    print(json.dumps({'gemini_key_present': ready, 'upstream_present': (ROOT / 'methods/agents/plotting_agent.py').is_file()}), flush=True)
    if not args.run:
        return
    if not ready:
        raise SystemExit('Set GEMINI_API_KEY in the local ignored .env; never put it in chat.')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path('C:/Users/leduc/Downloads') / ('M3_PaperOrchestra_' + stamp)
    output.mkdir(exist_ok=False)

    # Bound retry amplification for this one-figure trial, without editing upstream.
    import utils.paper_banana_utils as helpers
    helpers.call_gemini_with_contents = functools.partial(helpers.call_gemini_with_contents, max_retries=1)
    helpers.generate_image_with_gemini = functools.partial(helpers.generate_image_with_gemini, max_retries=1)
    original_visualizer = helpers.generate_figure_visuals

    def checked_visualizer(*positional, **named):
        (output / 'styled_description.txt').write_text(named['figure_description'], encoding='utf-8')
        value = original_visualizer(*positional, **named)
        if not value:
            raise RuntimeError('No image returned; stopping before additional critic/caption calls.')
        base64.b64decode(value, validate=True)
        return value

    helpers.generate_figure_visuals = checked_visualizer
    from methods.agents.plotting_agent import process_single_figure
    plan = dict(figure_id='M3', title='Image-specific prompt-visual alignment', plot_type='diagram', data_source='idea_sparse.md', aspect_ratio='16:9', objective='Draw the exact M3 two-stream mechanism in idea_sparse.md. Prioritize correct sources, concise labels, clear left-to-right flow and strictly straight/orthogonal arrows. Follow the supplied visual specification; omit a figure title in the image.')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    try:
        result = process_single_figure(plan, str(materials), model_name='gemini-3.5-flash', image_model_name='gemini-3-pro-image', max_critic_rounds=1)
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps(dict(status='INCOMPLETE', error_type=type(error).__name__, image_generated=False)), encoding='utf-8')
        raise
    data = base64.b64decode(result.pop('base64_img'), validate=True)
    from PIL import Image
    import io
    with Image.open(io.BytesIO(data)) as picture:
        picture.load()
        picture.save(output / 'M3_PaperOrchestra.png')
        size = picture.size
    (output / 'generation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt = dict(upstream_commit='d9bae997e793a015ed38bb1605be9eb58c00284f', writer_model='gemini-3.5-flash', image_model='gemini-3-pro-image', paperbanana_examples=False, max_critic_rounds=1, image_size=size, image_sha256=hashlib.sha256((output / 'M3_PaperOrchestra.png').read_bytes()).hexdigest(), research_models_executed=False)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('OUTPUT:', output, flush=True)


if __name__ == '__main__':
    main()
