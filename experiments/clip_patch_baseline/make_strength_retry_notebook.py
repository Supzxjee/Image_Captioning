"""Author cap-only V2 from the committed V1; never launch Kaggle/models."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE='57db104:experiments/clip_patch_baseline/CLIP_PATCH_SEMANTIC_STRENGTH_XE10_2GPU.ipynb'


def make():
    raw=subprocess.check_output(['git','show',SOURCE],cwd=ROOT);old=json.loads(raw);new=json.loads(raw)
    changes=0
    for cell in new['cells']:
        text=''.join(cell['source'])
        if cell['cell_type']=='code':
            if 'WALL_HOURS=10.5 ' in text:
                text=text.replace('WALL_HOURS=10.5 ','WALL_HOURS=11.5 ',1)
                text=text.replace('clip_patch_semantic_strength_xe10_v1','clip_patch_semantic_strength_xe10_v2')
                text=text.replace('cfg=dict(semantic=True,','cfg=dict(parent_wall_hours=WALL_HOURS,semantic=True,',1);changes+=1
            ast.parse(text)
        else:
            text=text.replace('Parent10.5h','Parent11.5h').replace('clip_patch_semantic_strength_xe10_v1_review.zip','clip_patch_semantic_strength_xe10_v2_review.zip')
            text+='\nV2 cap amendment: previous run passed GPU guards but stopped at0mainupdates because projection exceeded available cap by407seconds. Parent11.5h within user-selected own15h budget; same30code assets pinnedcc6a20e, no model/loss/schedule changes or margin removal. Fresh runtime restages /tmp; SaveVersion output does not include that cache. This is not automatic recovery. Check actual remaining quota before user launch; no billing/ETA guarantee.\n'
        cell['source']=text if isinstance(cell['source'],str) else text.splitlines(keepends=True)
    if changes!=1:raise ValueError('Expected one original parent-cap setup cell')
    target=HERE/'CLIP_PATCH_SEMANTIC_STRENGTH_XE10_V2_2GPU.ipynb';target.write_text(json.dumps(new,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    assert new['cells'][2:]==old['cells'][2:],'Runner/model execution must be unchanged'
    setup=''.join(new['cells'][1]['source']);original=''.join(old['cells'][1]['source'])
    expected=original.replace('WALL_HOURS=10.5 ','WALL_HOURS=11.5 ',1).replace('clip_patch_semantic_strength_xe10_v1','clip_patch_semantic_strength_xe10_v2').replace('cfg=dict(semantic=True,','cfg=dict(parent_wall_hours=WALL_HOURS,semantic=True,',1)
    assert setup==expected
    assert 11.5<15 and 33994.2317196553<33586.869474751+3600
    checks=dict(status='CAP_ONLY_STATIC_PASS_GPU_UNRUN',original_notebook=SOURCE,original_Git_blob_sha256=hashlib.sha256(raw).hexdigest(),new_notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        parent_hours_old=10.5,parent_hours_new=11.5,user_budget_hours_before_failed_run=15,actual_remaining_quota_unmeasured=True,
        setup_only_expected_changes=True,runner_and_execution_cells_unchanged=True,all_code_AST=True,model_code_revision='cc6a20ea7f7f3263a7a637100ab1166dd6ff6a1b',
        projection_guard_margin_unchanged=True,epochs_10_unchanged=True,previous_main_updates=0,previous_deficit_seconds=407.36224490429595,
        repeat_projection_extra_margin_seconds=33586.869474751+3600-33994.2317196553,neural_or_scorer_or_Kaggle_executed_locally=False)
    (HERE/'SEMANTIC_STRENGTH_V2_AUTHOR_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(target.name,'cap-only checks PASS; no GPU run.')

if __name__=='__main__':make()
