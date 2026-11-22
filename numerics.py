"""Diagnostics against measured naive logits; no assumed global error bound."""
import torch
from inference import logit_tolerance


def compare_logits(reference, cached):
    """Summarize finite, equally shaped (..., vocabulary) logits.

    If the naive top-two margin exceeds twice the measured maximum error in
    that row, its winner cannot change under that measured perturbation.
    This is an a posteriori check, not a guarantee for unseen prefixes.
    """
    if reference.shape != cached.shape or reference.ndim < 1 or reference.shape[-1] < 2 or reference.numel() == 0:
        raise ValueError("expected matching nonempty logits with at least two classes")
    if not reference.is_floating_point() or not cached.is_floating_point():
        raise ValueError("logits must be floating point")
    ref, actual = reference.detach().double(), cached.detach().to(reference.device).double()
    if not bool(torch.isfinite(ref).all() & torch.isfinite(actual).all()):
        raise ValueError("nonfinite logits")
    errors = (ref - actual).abs().amax(-1)
    top = ref.topk(2, dim=-1).values
    margins = top[..., 0] - top[..., 1]
    stable = margins > 2 * errors
    mismatch = ref.argmax(-1) != actual.argmax(-1)
    threshold = logit_tolerance(reference.dtype, ref.abs().max())
    return {"rows": errors.numel(), "max_abs_error": errors.max().item(),
            "historical_threshold": threshold,
            "within_historical_threshold": bool((errors <= threshold).all()),
            "greedy_mismatches": int(mismatch.sum()),
            "min_naive_margin": margins.min().item(),
            "rows_with_margin_over_twice_error": int(stable.sum()),
            "rows_without_margin_certificate": int((~stable).sum())}
