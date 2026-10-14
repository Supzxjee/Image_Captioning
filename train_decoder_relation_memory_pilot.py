"""Matched 512-image / two-epoch decoder adaptation; no evaluation-based tuning."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path
import torch
import torch.nn.functional as F
from captioning.checkpoints import load_checkpoint
from captioning.grounded_relations import align_grounded_relation_cache
from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
from captioning.relation_memory_pilot import filename_index, choose_training, action_entry, adapt_parameters
from captioning.tokenizer import CaptionTokenizer
from captioning.visual_cache import VisualCache, coco_image_id


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'dataset-json', 'prompt-cache', 'training-grounded-cache',
                 'selected', 'evaluation-grounded-cache', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--visual-cache', action='append', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a fresh pilot output directory')
    output.mkdir(parents=True)
    # Fixed a priori; no parameter search over the inspected 12-image result.
    seed, epochs, batch_size, image_count, lr = 42, 2, 16, 512, 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('visual_adapter') != 'qformer' or not saved.get('grounded_relation_alignment') or
            saved.get('grounded_relation_version', 'v1') != 'v1' or
            saved.get('decoder_relation_memory', False) or saved.get('prompt_conditioned_qformer') or
            saved.get('object_semantic_alignment') or saved.get('cascade_semantic_alignment') or saved.get('itc_weight', 0)):
        raise ValueError('This matched pilot requires the historical E3 V1 checkpoint without extra architectures')
    tokenizer = CaptionTokenizer([])
    vocabulary = saved['caption_word2idx']
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError('Caption vocabulary IDs must be contiguous')
    if any(vocabulary.get(k) != v for k, v in tokenizer.word2idx.items()):
        raise ValueError('Caption special IDs differ')
    tokenizer.word2idx = vocabulary
    tokenizer.idx2word = {value: key for key, value in vocabulary.items()}
    adapter_settings = dict(num_visual_queries=saved['num_visual_queries'], qformer_layers=saved['qformer_layers'])
    source_epoch = saved['epoch']
    del saved
    prompts = filename_index(torch.load(args.prompt_cache, map_location='cpu', weights_only=False)['data'])
    bundle = torch.load(args.training_grounded_cache, map_location='cpu', weights_only=False)
    if bundle.get('metadata', {}).get('inference_only') or bundle.get('metadata', {}).get('variant') != 'instance_pair_grounded_relation_v1':
        raise ValueError('Use the original full V1 training cache, not the selected diagnostic cache')
    relations = filename_index(bundle['data']); del bundle
    records = json.loads(Path(args.dataset_json).read_text(encoding='utf-8'))['images']
    selected = json.loads(Path(args.selected).read_text(encoding='utf-8'))
    excluded = set(selected)  # All 100 inspected pool images, not just the chosen 12.
    cohort = choose_training(records, prompts, relations, excluded, image_count, seed)
    names = [r['filename'] for r in cohort]
    relation_inputs = align_grounded_relation_cache({n: action_entry(relations[n]) for n in names}, names)
    assert not set(names) & excluded
    visual = VisualCache(args.visual_cache, id_key='coco_id')
    visual.require_ids([coco_image_id(n) for n in names], 'pilot training')
    protocol = dict(seed=seed, epochs=epochs, batch_size=batch_size, images=image_count, lr=lr,
        checkpoint_sha256=sha(args.checkpoint), source_epoch=source_epoch,
        dataset_sha256=sha(args.dataset_json), training_cache_sha256=sha(args.training_grounded_cache),
        prompt_sha256=sha(args.prompt_cache), selected_sha256=sha(args.selected),
        train_filenames=names, excluded_evaluation_filenames=sorted(excluded),
        notes='Caption CE only; heuristic action-positive training cache, not human ground truth. '
              'Same cohort/order/init/trainable parameter names; encoder/language path frozen. '
              'Dropout seed same but memory shapes differ. Evaluation is reused inspected development, '
              'not heldout scientific evidence. No early stopping, threshold/loss/epoch search.')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    try:
        for condition in ('legacy', 'direct'):
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            encoder = UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer',
                grounded_relation_alignment=True, **adapter_settings)
            model = ImageCaptioningModel(encoder, CaptionDecoder(len(vocabulary))).to(device)
            load_checkpoint(model, args.checkpoint, tokenizer)  # Identical strict source for both.
            encoder.decoder_relation_memory = condition == 'direct'
            parameters, parameter_names = adapt_parameters(model)
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)
            history = []
            directory = output/condition; directory.mkdir()
            for epoch in range(epochs):
                model.eval(); model.decoder.train()  # Frozen encoder/dropout stays deterministic.
                order = list(range(len(cohort))); random.Random(seed+epoch).shuffle(order)
                total, batches = 0., 0
                for start in range(0, len(order), batch_size):
                    batch = [cohort[i] for i in order[start:start+batch_size]]
                    image = torch.stack([torch.from_numpy(visual.read(coco_image_id(r['filename']))) for r in batch]).to(device)
                    prompt = torch.stack([prompts[r['filename']]['tokens'] for r in batch]).float().to(device)
                    mask = torch.stack([prompts[r['filename']]['mask'] for r in batch]).to(device)
                    entries = [relation_inputs[r['filename']] for r in batch]
                    kwargs = {argument: torch.stack([e[key] for e in entries]).to(device)
                        for argument,key in [('relation_tokens','tokens'),('negative_relation_tokens','negative_tokens'),
                            ('subject_boxes','subject_boxes'),('object_boxes','object_boxes'),
                            ('relation_confidences','confidences'),('relation_mask','mask')]}
                    kwargs['relation_tokens'] = kwargs['relation_tokens'].float()
                    kwargs['negative_relation_tokens'] = kwargs['negative_relation_tokens'].float()
                    with torch.no_grad(): memory, memory_mask = model.build_memory(image, prompt, mask, **kwargs)
                    # Keep every available first-five caption; repeat memory consistently.
                    encoded, mapping = [], []
                    for index, record in enumerate(batch):
                        for sentence in record['sentences'][:5]:
                            encoded.append(tokenizer.encode(sentence['raw'])[0]); mapping.append(index)
                    target = torch.stack(encoded).to(device)
                    indices = torch.tensor(mapping, device=device)
                    logits = model.decoder(target[:,:-1], memory[indices],
                        tgt_key_padding_mask=target[:,:-1] == tokenizer.pad_idx,
                        memory_key_padding_mask=memory_mask[indices])
                    loss = F.cross_entropy(logits.reshape(-1,len(vocabulary)), target[:,1:].reshape(-1), ignore_index=tokenizer.pad_idx)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite caption loss')
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1); optimizer.step()
                    total += float(loss.detach()); batches += 1
                    if batches == 1 or batches % 8 == 0:
                        print(f'{condition} epoch {epoch+1}/{epochs} batch {batches}/32 CE={float(loss.detach()):.4f}', flush=True)
                history.append(dict(epoch=epoch+1, caption_loss=total/batches, batches=batches))
            model.eval()
            checkpoint = dict(epoch=epochs, visual_adapter='qformer', **adapter_settings,
                grounded_relation_alignment=True, grounded_relation_version='v1',
                decoder_relation_memory=condition=='direct', caption_word2idx=vocabulary,
                encoder_state_dict=model.encoder.state_dict(), decoder_state_dict=model.decoder.state_dict(),
                pilot_source_epoch=source_epoch, pilot_source_checkpoint_sha256=protocol['checkpoint_sha256'],
                relation_alignment_weight=0., relation_alignment_margin=.2)
            path = directory/'adapted_epoch_2.pth'; torch.save(checkpoint,path)
            (directory/'training.json').write_text(json.dumps(dict(history=history,trainable_names=parameter_names,
                trainable_parameters=sum(p.numel() for p in parameters),checkpoint_sha256=sha(path)),indent=2),encoding='utf-8')
            del checkpoint, model, encoder, optimizer, parameters
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        visual.close()
    del prompts, relations, records, cohort, relation_inputs
    for condition in ('legacy','direct'):
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('probe_selected_caption_grounding.py')),
            '--checkpoint',str(output/condition/'adapted_epoch_2.pth'),'--selected',args.selected,
            '--grounded-cache',args.evaluation_grounded_cache,'--prompt-cache',args.prompt_cache,
            *[part for path in args.visual_cache for part in ('--visual-cache',path)],
            '--output',str(output/condition/'evaluation_12'),'--sample-size','12'],check=True)
    with zipfile.ZipFile(str(output)+'_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob('*.json'):
            archive.write(path,path.relative_to(output))
    print('Review ZIP:',str(output)+'_review.zip',flush=True)
    print('Both adapted checkpoints retained separately in output; excluded from review ZIP.',flush=True)


if __name__ == '__main__': main()
