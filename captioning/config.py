"""Architecture constants and runtime settings; no datasets or models loaded here."""
from dataclasses import dataclass, field, replace
from pathlib import Path

MODEL_NAME = 'clip'
EMBED_DIM = 512
MAX_PROMPT_LEN = 20
MAX_CAPTION_LEN = 30
NUM_HEADS = 8

@dataclass
class Config:
    mode: str = 'train'
    epochs: int = 10
    lr: float = 1e-4
    checkpoint: str = ''
    split: str = 'val'
    limit: int = 0
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    base_path: str = '/kaggle/input/datasets/vuthetam/mscoco-2014/images'
    dataset_json_path: str = '/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json'
    prompt_cache_path: str = '/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt'
    work_dir: Path = Path('/kaggle/working')
    experiment_name: str = 'h1_2_gated_prompt_to_visual_crossattn'
    device: str = 'cpu'
    test_after_train: bool = False
    test_visual_source: str = 'same'
    predictions: str = ''
    ground_truth: str = ''
    metrics_output: str = ''
    visual_cache_id_key: str = 'coco_id'
    visual_preprocessing: str = 'bilinear'
    visual_precision: str = 'fp32'
    visual_cache: list[str] = field(default_factory=list)
    region_targets_path: str = ''
    alignment_weight: float = 0.0
    alignment_temperature: float = 0.07
    max_regions: int = 10
    max_train_batches: int = 0
    visual_adapter: str = 'direct'
    num_visual_queries: int = 32
    qformer_layers: int = 2
    caption_embedding_cache_path: str = ''
    itc_weight: float = 0.0
    itc_temperature: float = 0.07

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        if self.test_visual_source not in {'same', 'images'}:
            raise ValueError('Invalid test visual source.')
        if self.visual_cache_id_key not in {'coco_id', 'eval_id', 'karpathy_id'}:
            raise ValueError('Invalid visual cache ID key.')
        if self.visual_preprocessing not in {'bilinear', 'bicubic'} or self.visual_precision not in {'fp32', 'amp-fp16'}:
            raise ValueError('Invalid visual preprocessing or precision.')
        if self.mode not in {'train', 'evaluate', 'predict', 'metrics', 'verify-cache'} or self.split not in {'val', 'test'}:
            raise ValueError('Invalid mode or evaluation split.')
        if self.test_after_train and self.mode != 'train':
            raise ValueError('--test-after-train is only valid with train mode.')
        if self.mode == 'metrics' and not (self.predictions and self.ground_truth):
            raise ValueError('metrics mode requires --predictions and --ground-truth.')
        if self.mode == 'verify-cache' and not self.visual_cache:
            raise ValueError('verify-cache requires --visual-cache.')
        if self.mode == 'train' and self.alignment_weight > 0 and not self.region_targets_path:
            raise ValueError('Positive alignment weight requires --region-targets-path.')
        if self.alignment_weight < 0 or self.alignment_temperature <= 0 or self.max_regions <= 0:
            raise ValueError('Alignment weight must be nonnegative; temperature/max-regions positive.')
        if self.max_train_batches < 0:
            raise ValueError('max-train-batches must be nonnegative.')
        if self.visual_adapter not in {'direct', 'qformer'}:
            raise ValueError('visual-adapter must be direct or qformer.')
        if self.num_visual_queries <= 0 or self.qformer_layers <= 0:
            raise ValueError('num-visual-queries and qformer-layers must be positive.')
        if self.visual_adapter == 'qformer' and self.alignment_weight > 0:
            raise ValueError('The first Q-Former ablation does not combine region alignment loss.')
        if self.itc_weight < 0 or self.itc_temperature <= 0:
            raise ValueError('ITC weight must be nonnegative and temperature positive.')
        if self.itc_weight > 0 and self.visual_adapter != 'qformer':
            raise ValueError('ITC currently requires visual-adapter=qformer.')
        if self.mode == 'train' and self.itc_weight > 0 and not self.caption_embedding_cache_path:
            raise ValueError('Positive ITC weight requires --caption-embedding-cache-path.')
        if self.epochs <= 0 or self.lr <= 0 or self.limit < 0 or self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError('Epochs, learning rate and batch size must be positive; limit/workers nonnegative.')

    @property
    def checkpoint_dir(self):
        return self.work_dir / self.experiment_name / 'checkpoints'

    @property
    def eval_dir(self):
        return self.work_dir / self.experiment_name / 'evaluation'


def post_train_test_config(config):
    """Use the newly trained final checkpoint, not the warm-start input."""
    return replace(config, mode='evaluate', checkpoint='', split='test', limit=0,
                   test_after_train=False,
                   visual_cache=[] if config.test_visual_source == 'images' else list(config.visual_cache))
