"""CLI parsing is independent of PyTorch, so --help works before installation."""
import argparse
from .config import Config

def parse_args(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser(description='H1.2-G image captioning on Kaggle')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict', 'metrics', 'verify-cache'], default='train')
    parser.add_argument('--test-after-train', action='store_true', help='After training, generate full test captions and compute metrics.')
    parser.add_argument('--test-visual-source', choices=['same', 'images'], default='same',
                        help='For test-after-train: images runs CLIP directly when test features are absent.')
    parser.add_argument('--predictions', default='', help='Saved predictions JSON for metrics mode.')
    parser.add_argument('--ground-truth', default='', help='Saved COCO ground truth JSON for metrics mode.')
    parser.add_argument('--metrics-output', default='', help='Metrics JSON output path.')
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--lr', type=float, default=defaults.lr)
    parser.add_argument('--visual-cache-id-key', choices=['coco_id', 'eval_id', 'karpathy_id'], default='coco_id',
                        help='coco_id = filename ID; eval_id = zero-based row in dataset_coco.json. karpathy_id = imgid field. Verify before use.')
    parser.add_argument('--visual-preprocessing', choices=['bilinear', 'bicubic'], default='bilinear')
    parser.add_argument('--visual-precision', choices=['fp32', 'amp-fp16'], default='fp32')
    parser.add_argument('--visual-cache', action='append', default=[],
                        help='HDF5 file or directory; repeat to use multiple files. Omit to run CLIP directly.')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--limit', type=int, default=0, help='0 = full split; positive = first N images')
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--num-workers', type=int, default=defaults.num_workers)
    parser.add_argument('--base-path', default=defaults.base_path)
    parser.add_argument('--dataset-json-path', default=defaults.dataset_json_path)
    parser.add_argument('--prompt-cache-path', default=defaults.prompt_cache_path)
    parser.add_argument('--work-dir', default=str(defaults.work_dir))
    parser.add_argument('--experiment-name', default=defaults.experiment_name)
    parser.add_argument('--region-targets-path', default='',
                        help='Prebuilt normalized YOLO boxes and CLIP label prototypes for region loss.')
    parser.add_argument('--alignment-weight', type=float, default=0.0,
                        help='Lambda for object-region classification loss; 0 keeps the baseline unchanged.')
    parser.add_argument('--alignment-temperature', type=float, default=0.07)
    parser.add_argument('--max-regions', type=int, default=10)
    parser.add_argument('--max-train-batches', type=int, default=0,
                        help='Smoke-test cap per epoch; 0 uses the full training loader.')
    return Config(**vars(parser.parse_args(argv)))

def main(argv=None):
    config = parse_args(argv)
    if config.mode == 'metrics':
        from .metrics import score_files
        score_files(config.predictions, config.ground_truth, config.metrics_output or None)
        return
    from .pipeline import run
    run(config)
