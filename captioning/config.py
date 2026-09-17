"""Architecture constants and runtime settings; no datasets or models loaded here."""
from dataclasses import dataclass, field
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
    visual_cache: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        if self.mode not in {'train', 'evaluate', 'verify-cache'} or self.split not in {'val', 'test'}:
            raise ValueError('Invalid mode or evaluation split.')
        if self.mode == 'verify-cache' and not self.visual_cache:
            raise ValueError('verify-cache requires --visual-cache.')
        if self.epochs <= 0 or self.lr <= 0 or self.limit < 0 or self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError('Epochs, learning rate and batch size must be positive; limit/workers nonnegative.')

    @property
    def checkpoint_dir(self):
        return self.work_dir / self.experiment_name / 'checkpoints'

    @property
    def eval_dir(self):
        return self.work_dir / self.experiment_name / 'evaluation'
