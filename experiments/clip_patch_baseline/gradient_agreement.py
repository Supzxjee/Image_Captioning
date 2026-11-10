"""Scale-aware FP32 gradient diagnostics; NumPy only, no neural execution."""
import numpy as np


def gradient_agreement(actual, reference):
    a = np.asarray(actual, dtype=np.float64); b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Gradient shape/finite contract')
    delta = a-b; absolute = float(np.abs(delta).max()); norm = float(np.linalg.norm(b.ravel()))
    reference_max = float(np.abs(b).max()); error_norm = float(np.linalg.norm(delta.ravel()))
    relative = error_norm/max(norm, 1e-30)
    threshold = 3e-7 if norm <= 1e-7 else 3e-7 + .005*reference_max
    at = tuple(int(x) for x in np.unravel_index(np.abs(delta).argmax(), a.shape))
    return dict(passed=absolute<=threshold and (norm<=1e-7 or relative<=.005),
                reference_norm=norm, reference_max_abs=reference_max, error_norm=error_norm,
                relative_l2=relative, max_abs_error=absolute, max_abs_limit=threshold,
                scale_relative_max_error=absolute/max(reference_max, 1e-30), max_error_index=at,
                actual_at_max=float(a[at]), reference_at_max=float(b[at]))
