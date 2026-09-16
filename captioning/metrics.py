import json

def export_ground_truth(dataframe, output_path):
    images, annotations = [], []
    annotation_id = 0
    for _, row in dataframe.iterrows():
        image_id = int(row['eval_id'])
        images.append({'id': image_id})
        for caption in row['captions']:
            annotations.append({'id': annotation_id, 'image_id': image_id, 'caption': caption})
            annotation_id += 1

    payload = {
        'info': {},
        'licenses': [],
        'type': 'captions',
        'images': images,
        'annotations': annotations,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

def compute_coco_metrics(predictions_path, ground_truth_path):
    from pycocotools.coco import COCO
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    coco_gt = COCO(str(ground_truth_path))
    coco_res = coco_gt.loadRes(str(predictions_path))
    image_ids = coco_res.getImgIds()

    gts = {image_id: coco_gt.imgToAnns[image_id] for image_id in image_ids}
    res = {image_id: coco_res.imgToAnns[image_id] for image_id in image_ids}

    ptb_tokenizer = PTBTokenizer()
    gts = ptb_tokenizer.tokenize(gts)
    res = ptb_tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
        (Meteor(), 'METEOR'),
        (Rouge(), 'ROUGE_L'),
        (Cider(), 'CIDEr'),
    ]

    metrics = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(names, list):
            metrics.update({name: float(value) for name, value in zip(names, score)})
        else:
            metrics[names] = float(score)
    return metrics
