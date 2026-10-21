"""Source/header audit only. Never import reference modules or deserialize pickle."""
import ast
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile

HERE=Path(__file__).resolve().parent
REVIEW=HERE/'review'
REF=REVIEW/'reference'


def prototype(name,expected_shape):
    path=REF/'coco_caption'/name
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        member=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        ops=list(pickletools.genops(z.read(member)))
        first=next(i for i,(op,_,_) in enumerate(ops) if op.name=='TUPLE2')
        shape=[ops[first-2][1],ops[first-1][1]]
        assert shape==expected_shape
        assert any(op.name=='GLOBAL' and arg=='torch FloatStorage' for op,arg,_ in ops)
        storage=next(n for n in z.namelist() if '/data/' in n)
        assert z.getinfo(storage).file_size==shape[0]*shape[1]*4
    return dict(filename=name,shape=shape,dtype='float32 from FloatStorage metadata',
        bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validation='ZIP CRC + pickle opcode metadata + storage length; no pickle execution or torch.load')


def linear(i,o,bias=True):return i*o+(o if bias else 0)


def run():
    provenance=json.loads((REVIEW/'provenance.json').read_text(encoding='utf-8'))
    assert provenance['commit']=='0cc570ba7c50ef08c46a21439fe73ef0a5df97ab'
    tree=json.loads((REVIEW/'tree.json').read_text(encoding='utf-8'))
    assert not tree.get('truncated')
    paths={r['path'] for r in tree['tree'] if r['type']=='blob'}
    nodes={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    for row in provenance['files']:
        raw=(REF/row['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==nodes[row['path']]['sha']
        if row['path'].endswith('.py'):ast.parse(raw.decode('utf-8'))
    banks=[prototype('coco_nouns_emb.pth',[4764,1024]),prototype('coco_verbs_emb.pth',[975,1024])]
    src=(REF/'models/caption/csa_net.py').read_text(encoding='utf-8')
    assert all(s in src for s in ['object_semantic_agg_k','relation_semantic_agg_k',
        'object_encoder_layer0','object_encoder_layer1','relation_encoder_layer2'])
    assert 'models/detection/det_module.py' not in paths
    assert 'models.detection.det_module' in (REF/'models/caption/detector.py').read_text(encoding='utf-8')
    d,f=512,2048
    layer=4*d*d+4*d + 2*(2*d) + linear(d,f)+linear(f,d)
    aggregation=linear(1024,1024)+linear(1024,150)+linear(1024,d)
    stage=2*layer+linear(2*d,1,False)+linear(d,d)
    total=2*aggregation+3*stage
    osa=aggregation+2*stage
    return dict(status='SOURCE_FEASIBILITY_PASS_NUMERICAL_PARITY_AND_PILOT_NOT_DONE',date='2026-10-22',
        reference=provenance,prototype_banks=banks,
        core=dict(input='reg_feat B x N x512; default official detector N150',
            semantic_input=['4764x1024 nouns','975x1024 verbs'],aggregated_slots=150,
            stages=['object0','object1','relation2'],output=['B x N x512 object','B x N x512 relation','B x1x1xN rel_mask'],
            n_layers_argument='Exposed but forward is hardcoded as two object plus one relation stage',
            requires_detector_import=False,requires_new_regionclip_inference_if_banks_reused=False),
        resource_estimate=dict(full_core_trainable_parameters=total,osa_two_stage_parameters=osa,
            full_core_fp32_parameters_grad_adam_bytes=total*16,
            osa_fp32_parameters_grad_adam_bytes=osa*16,prototype_file_bytes=sum(b['bytes'] for b in banks),
            pilot_gpu_hours='Unknown: no benchmark/model execution; cannot infer original two-A800 recipe fits T4x2'),
        full_system=dict(ready=False,missing_import='models/detection/det_module.py',
            detector_checkpoint='coco_caption/detector_checkpoint_4ds.pth referenced, absent in pinned tree; external availability not established',
            full_caption_checkpoint='No downloadable full-caption checkpoint identified in inspected README/tree',
            feature_cache='all_splits.h5 referenced, absent in pinned tree; no matching CSA region cache verified locally',
            original_recipe='Paper p7: SwinB/deformableDETR,2A800,batch32,15XE+30SC,3epochXE warmup,beam5; not current pilot recipe'),
        discrepancies=[
            'Paper Eq6 normalized cosine; released alignment uses raw dot products without F.normalize',
            'Released code uses region-pooling temperature0.1 and prototype-weighting temperature0.05; Eq8 writes softmax(R) without second scale',
            'Paper Eq11 weights self branch by G; code weights cross branch by response_gate and adds out_proj; choose explicit reference, not silent fixes',
            'Released relation mask equals sum(features)==1; validity meaning unexplained; cannot assume ordinary padding mask',
            'Parallel decoder alpha2 uses fc_alpha1 although fc_alpha2 is defined; code also divides sum by sqrt2',
            'Paper/config/README differ in warmup,LRate/batch/freeze recipe; pin one declared variant',
            'Released RL uses beam candidate mean reward baseline, not the current greedy SCST + CE anchor + object reward'
        ],
        inheritance_options=[
            dict(name='OSA component adaptation',status='Smallest component option; not itself a relation-alignment solution',
                preserved='Original noun bank,semantic aggregation,weighted alignment,two object fusion stages',
                changed='Input QFormer32queries instead of detector150regions; current decoder/legacy/recipe retained; no relation stage',
                truthful_label='Kế thừa khối Object Semantic Alignment của CSA-Net trên nền Q-Former, không tái hiện toànCSA-Net'),
            dict(name='Full CSA-core adaptation',status='Source-level feasible, decoder integration undecided',
                preserved='All2object+1relation stages and both original semantic banks',
                changed='QFormerquery representation; feeding both outputs to existing single-memory decoder differs from original parallel adaptive decoder',
                truthful_label='Kế thừa lõi CSA, khai báo đầu vào/readout thay đổi; không so trực tiếp điểm bài gốc'),
            dict(name='Full-system reproduction',status='Not ready from current release/assets',
                preserved='Requires detector/region semantics,dual-streamdecoder,originalvocabulary/features/trainingrecipe',
                changed='Cannot substitute CLIPpatches or UPT1024vectors and claim same input',truthful_label='Only call full reproduction after dependencies and recipe verified')
        ],
        E2=dict(status='CSA-inspired, not faithful OSA/CSA reproduction',
            local_source='captioning/models.py: CascadeSemanticAlignment',
            differences=['Per-image CLIP prompt/object tokens, not frozen global RegionCLIP noun/verb banks',
                'No learned No→K semantic aggregation; selector pools object context then scores prompt tokens',
                'Single query-to-semantic attention with512dim residual gate, not2object+1relation SA/CA/FFN scalarfusion stages',
                'No separate object/relation outputs or original dual-stream decoder'],
            scope='Inspected current source and TRAIN_E2_RELATION_V2_1.md, not all historic checkpoint artifacts'),
        next_gate='Choose component and reference convention with Astra/user; then verify tensor outputs+input/parameter gradients against released code on synthetic tensors, buffer/mask handling and exact baseline bypass BEFORE any GPU pilot.',
        contribution='Inheritance is a research foundation/control, not novelty. New claim must address a specific limitation and beat matched inherited baseline, with factual evaluation.',
        model_or_reference_code_executed=False,numerical_parity_tested=False,training_or_inference_executed=False)


if __name__=='__main__':
    result=run()
    (HERE/'CSA_INHERITANCE_FEASIBILITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','resource_estimate','full_system','E2']},ensure_ascii=False,indent=2))
