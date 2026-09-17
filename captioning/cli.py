"""CLI parsing is independent of PyTorch, so --help works before installation."""
import argparse
from .config import Config

def parse_args(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser(description='H1.2-G image captioning on Kaggle')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'verify-cache'], default='train')
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--lr', type=float, default=defaults.lr)
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
    return Config(**vars(parser.parse_args(argv)))

def main(argv=None):
    config = parse_args(argv)
    from .pipeline import run
    run(config)
