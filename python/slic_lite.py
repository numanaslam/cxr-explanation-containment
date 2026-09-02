"""Self-contained SLIC superpixels (numpy + scipy only).

Drop-in for the one scikit-image function this pipeline uses, for machines
where scikit-image cannot be installed (restricted networks). Implements
Achanta et al. (2012) SLIC for 2-D grayscale images with the skimage-style
`mask=` restriction: cluster centers are seeded on a grid whose step is
computed from the MASKED area, distances use the skimage weighting
D^2 = dc^2 + (ds/S)^2 * compactness^2, and every returned label lies inside
the mask (0 outside). Fully deterministic: grid init + fixed-point iteration,
no randomness.

Numbers from this segmenter are internally consistent but not bit-identical
to scikit-image's C implementation; explain.py stamps which segmenter
produced each mask store, and one segmenter must be used for a whole study.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def slic(image, n_segments=100, compactness=10.0, channel_axis=None,
         mask=None, start_label=1, max_num_iter=10):
    g = np.asarray(image, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("slic_lite supports 2-D grayscale images only")
    H, W = g.shape
    m = np.ones((H, W), bool) if mask is None else np.asarray(mask, bool)
    area = int(m.sum())
    if area == 0:
        return np.zeros((H, W), int)

    # ---- seed centers on a grid sized so ~n_segments fall inside the mask ----
    S = max(2, int(round(np.sqrt(area / max(n_segments, 1)))))
    cy = np.arange(S // 2, H, S)
    cx = np.arange(S // 2, W, S)
    centers = [(float(y), float(x), g[y, x])
               for y in cy for x in cx if m[y, x]]
    if not centers:                       # mask thinner than the grid step
        ys, xs = np.nonzero(m)
        k = max(1, min(n_segments, area))
        idx = np.linspace(0, len(ys) - 1, k).astype(int)
        centers = [(float(ys[i]), float(xs[i]), g[ys[i], xs[i]]) for i in idx]
    C = np.array(centers)                 # (K, 3): y, x, intensity

    yy, xx = np.mgrid[0:H, 0:W]
    lab = np.zeros((H, W), int)           # 0 = unassigned / outside mask
    m2 = float(compactness) ** 2

    for _ in range(max_num_iter):
        best = np.full((H, W), np.inf)
        lab.fill(0)
        for k, (y0, x0, i0) in enumerate(C, 1):
            y1, y2 = max(0, int(y0) - S), min(H, int(y0) + S + 1)
            x1, x2 = max(0, int(x0) - S), min(W, int(x0) + S + 1)
            win = m[y1:y2, x1:x2]
            dc2 = (g[y1:y2, x1:x2] - i0) ** 2
            ds2 = ((yy[y1:y2, x1:x2] - y0) ** 2 + (xx[y1:y2, x1:x2] - x0) ** 2)
            d = dc2 + ds2 / (S * S) * m2
            upd = win & (d < best[y1:y2, x1:x2])
            best[y1:y2, x1:x2][upd] = d[upd]
            lab[y1:y2, x1:x2][upd] = k
        # update centers (empty clusters keep their position)
        newC = C.copy()
        for k in range(1, len(C) + 1):
            sel = lab == k
            if sel.any():
                newC[k - 1] = [yy[sel].mean(), xx[sel].mean(), g[sel].mean()]
        if np.abs(newC - C).max() < 0.5:
            C = newC
            break
        C = newC

    # ---- connectivity: keep each label's largest component ----
    out = np.zeros((H, W), int)
    nxt = 1
    for k in range(1, len(C) + 1):
        comp, nc = ndimage.label(lab == k)
        if nc == 0:
            continue
        if nc > 1:
            sizes = ndimage.sum(np.ones_like(comp), comp, range(1, nc + 1))
            comp = comp == (1 + int(np.argmax(sizes)))
        else:
            comp = comp > 0
        out[comp] = nxt
        nxt += 1

    # ---- absorb orphaned in-mask pixels into the nearest labeled pixel ----
    hole = m & (out == 0)
    if hole.any():
        idx = ndimage.distance_transform_edt(out == 0, return_distances=False,
                                             return_indices=True)
        out[hole] = out[tuple(i[hole] for i in idx)]
    out *= m

    # compact final labels to 1..K, then apply start_label offset
    u = np.unique(out[out > 0])
    remap = np.zeros(out.max() + 1, int)
    remap[u] = np.arange(1, len(u) + 1)
    out = remap[out]
    if start_label != 1:
        out[out > 0] += start_label - 1
    return out
