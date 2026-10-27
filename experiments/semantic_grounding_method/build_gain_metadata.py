"""Package previously audited metadata, never a neural runtime."""
import gzip
import json
from pathlib import Path
from gain_data import build_metadata,identity,save,sha


def main():
    here=Path(__file__).resolve().parent;root=here.parents[1]
    cached=json.loads((root/'experiments/stale_prompt_pilot/review/C_loaded.json').read_text(encoding='utf-8'))
    protocol=json.loads(cached['texts']['protocol.json'])
    meta=build_metadata(root/'experiments/coco_grounding_feasibility/review/noun_box_train10k_v1_1/dataset',
        protocol['groups'],protocol['word2idx'],json.loads(cached['texts']['A_predictions.json']))
    dest=here/'data/gain_metadata.json.gz';dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(gzip.compress(json.dumps(meta,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    record=dict(metadata_sha256=sha(dest),bytes=dest.stat().st_size,coverage=meta['coverage'],
        training_ids_sha256=identity([r['coco_id'] for r in meta['groups']['train']]),
        dev_ids_sha256=identity([r['coco_id'] for r in meta['groups']['val']]),
        baseline_predictions_sha256=identity(meta['baseline']),word2idx_sha256=identity(meta['word2idx']),
        local_neural_execution=False)
    save(here/'METADATA_PACKAGE.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
