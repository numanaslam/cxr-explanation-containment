"""Geometry-matched and mask-swap nulls (model-independent core).

Three chance baselines for explanation containment, each destroying only the
explanation's *alignment* with the anatomy while preserving stated geometry:

  rotation_null      preserves area, shape, distance-from-centre   (Phase 5, primary)
  translation_null   preserves area, shape (not eccentricity)      (control on the control)
  maskswap_null      preserves REAL explanation geometry; scores A_i against a lung
                     mask B_j from a different image                (JMI Concern 6/7)

The mask-swap null answers the reviewer's point that rotation rewards axis
alignment with the horizontally-elongated central lung field: swapping real masks
across images has no rotation artefact, so agreement between the two nulls shows
the conclusion is not null-specific.

All functions take/return boolean masks. Deterministic given `rng`.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import rotate as _rotate
from containment import precision, reference_fraction


def rotation_null(A, rng, exclude_deg=30.0):
    """Rotate A about the image centre by a random angle in
    [30,150] ∪ [210,330] deg. Preserves area, shape, radial distance; destroys
    anatomical alignment. Angles near 0 leave A in place; near 180 map the
    bilaterally symmetric lung field onto itself."""
    A = np.asarray(A).astype(bool)
    if rng.random() < 0.5:
        ang = exclude_deg + (180 - 2 * exclude_deg) * rng.random()          # 30..150
    else:
        ang = 180 + exclude_deg + (180 - 2 * exclude_deg) * rng.random()    # 210..330
    r = _rotate(A.astype(float), ang, reshape=False, order=0, mode="constant", cval=0.0)
    return r > 0.5


def translation_null(A, rng):
    """Circular shift: preserves area exactly and shape, not distance-from-centre."""
    A = np.asarray(A).astype(bool)
    dy = int(rng.integers(0, A.shape[0]))
    dx = int(rng.integers(0, A.shape[1]))
    return np.roll(A, (dy, dx), axis=(0, 1))


def lift_geometry(A, B, n_draws=100, rng=None, kind="rotation"):
    """Geometry-matched lift = P(A) - mean_j P(null_j), and the empirical
    p-value = fraction of null draws reaching the observed precision.

    kind: 'rotation' | 'translation'. Returns (lift, mean_null_precision, p_emp).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs = precision(A, B)
    fn = rotation_null if kind == "rotation" else translation_null
    ps = np.empty(n_draws)
    for j in range(n_draws):
        ps[j] = precision(fn(A, rng), B)
    mean_null = float(np.nanmean(ps))
    p_emp = float((1 + np.sum(ps >= obs)) / (n_draws + 1))
    return obs - mean_null, mean_null, p_emp


def lift_maskswap(A, B_self, B_others, n_draws=100, rng=None):
    """Mask-swap null: score A against lung masks from OTHER images.

    A          : explanation mask for image i (already at the working size)
    B_self     : image i's own lung mask (for the observed precision)
    B_others   : sequence of other images' lung masks, each same HxW as A
    Returns (lift_vs_swap, mean_swap_precision, p_emp).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    A = np.asarray(A).astype(bool)
    obs = precision(A, B_self)
    idx = rng.integers(0, len(B_others), size=n_draws)
    ps = np.array([precision(A, np.asarray(B_others[k]).astype(bool)) for k in idx])
    mean_swap = float(np.nanmean(ps))
    p_emp = float((1 + np.sum(ps >= obs)) / (n_draws + 1))
    return obs - mean_swap, mean_swap, p_emp


def interpret_negative_lift(lift_geom: float) -> str:
    """A negative geometry-matched lift means A lands inside the reference LESS
    often than rotated copies of itself -- i.e. the explanation is placed
    *systematically away* from the anatomy. Under masked-ROI inputs this is the
    signature of top-K segments concentrating on background (JMI Concern 6)."""
    if np.isnan(lift_geom):
        return "undefined (empty mask)"
    if lift_geom < -0.02:
        return ("NEGATIVE: explanation sits outside the reference more than its own "
                "rotations -- consistent with background-concentrated top-K, not anatomy")
    if lift_geom > 0.02:
        return "positive: anatomical specificity beyond centrality"
    return "~0: no specificity beyond centrality (centrality fully explains the lift)"
