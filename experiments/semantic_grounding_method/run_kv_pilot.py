"""CLI dispatcher; model code loads only in the explicitly user-run worker."""
import argparse
import json
from pathlib import Path
from kv_pilot_data import digest,prepare,ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('prepare','worker','evaluate'))
    p.add_argument('--config',type=Path,required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_bytes())
    for name,expected in cfg['source_assets'].items():
        if digest(ROOT/name) != expected:raise ValueError('Changed source '+name)
    if args.mode == 'prepare':prepare(cfg)
    elif args.mode == 'worker':
        from kv_pilot_training import worker
        worker(cfg)
    else:
        from kv_pilot_evaluate import evaluate
        evaluate(cfg)


if __name__ == '__main__':main()
