"""Kaggle-only synthetic tensor tests. No pretrained checkpoint is loaded."""
from captioning.scst_runtime import configure_strict_runtime
from utility_intervention import synthetic_checks

if __name__=='__main__':
    configure_strict_runtime(True)
    synthetic_checks()
