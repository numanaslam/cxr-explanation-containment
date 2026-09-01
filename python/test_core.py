"""Unit tests for the model-independent containment/null core.

Runs with only numpy+scipy -- no model, no data, no GPU -- so correctness is
established in the sandbox before the code ever touches real explanations.
"""
import numpy as np
import containment as C
import nulls as N


def lung_mask(h=128, w=128):
    """Two central ellipses ~ a lung field: horizontally elongated, central."""
    yy, xx = np.mgrid[0:h, 0:w]
    def ell(cy, cx, ry, rx):
        return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    return ell(h * 0.5, w * 0.36, h * 0.30, w * 0.14) | ell(h * 0.5, w * 0.64, h * 0.30, w * 0.14)


def central_square(h=128, w=128, s=20):
    m = np.zeros((h, w), bool)
    m[h // 2 - s // 2:h // 2 + s // 2, w // 2 - s // 2:w // 2 + s // 2] = True
    return m


def run():
    ok = 0
    B = lung_mask()

    # 1. precision/recall/iou on a hand-verifiable fully-contained mask
    A = np.zeros_like(B); A[60:68, 44:52] = True      # small block inside left lung
    assert (A & B).sum() == A.sum(), "test block must be fully inside the lung"
    assert abs(C.precision(A, B) - 1.0) < 1e-9
    assert abs(C.iou(A, B) - (A & B).sum() / (A | B).sum()) < 1e-9
    print(f"[1] precision={C.precision(A,B):.3f} recall={C.recall(A,B):.4f} iou={C.iou(A,B):.4f}  OK"); ok += 1

    # 2. analytic lift = precision - lung fraction
    lf = C.reference_fraction(B)
    assert abs(C.lift_analytic(A, B) - (1.0 - lf)) < 1e-9
    print(f"[2] lung fraction={lf:.4f}  analytic lift={C.lift_analytic(A,B):+.3f}  OK"); ok += 1

    # 3. THE centrality demonstration: a central square scores high ANALYTIC lift
    #    but ~0 GEOMETRY-MATCHED lift, because its rotations stay central too.
    sq = central_square()
    rng = np.random.default_rng(0)
    la = C.lift_analytic(sq, B)
    lg, mn, pe = N.lift_geometry(sq, B, n_draws=200, rng=rng, kind="rotation")
    print(f"[3] central square: analytic {la:+.3f} | rotation-null lift {lg:+.3f} "
          f"(null P={mn:.3f}, p_emp={pe:.2f})")
    assert la > 0.15, "central square should beat the uniform baseline"
    assert lg < la - 0.10, "rotation null must absorb most of the central-square lift"
    print("    -> rotation null absorbs the centrality  OK"); ok += 1

    # 4. rotation preserves area (order-0, 90 deg exact); translation preserves it exactly
    r90 = N._rotate(sq.astype(float), 90, reshape=False, order=0) > 0.5
    assert abs(r90.sum() - sq.sum()) / sq.sum() < 0.02, "90deg rotation ~area-preserving"
    tr = N.translation_null(sq, np.random.default_rng(1))
    assert tr.sum() == sq.sum(), "translation preserves area exactly"
    print(f"[4] area: obs={sq.sum()} rot90={r90.sum()} shift={tr.sum()}  OK"); ok += 1

    # 5. negative geometry lift when the explanation sits OFF the anatomy (corners)
    off = np.zeros_like(B); off[:18, :18] = True; off[:18, -18:] = True
    lg_off, _, _ = N.lift_geometry(off, B, n_draws=200, rng=np.random.default_rng(2))
    print(f"[5] off-anatomy mask: geometry-matched lift {lg_off:+.3f} -> "
          f"{N.interpret_negative_lift(lg_off)}")
    assert lg_off < 0.02, "off-anatomy explanation should not show positive specificity"
    print("    OK"); ok += 1

    # 6. mask-swap null: A_i scored against OTHER images' lung masks
    others = [np.roll(B, (int(d), int(-d)), axis=(0, 1)) for d in range(-10, 11) if d]
    lg_sw, mn_sw, pe_sw = N.lift_maskswap(A, B, others, n_draws=len(others),
                                          rng=np.random.default_rng(3))
    print(f"[6] mask-swap: obs P={C.precision(A,B):.3f} | swap-null P={mn_sw:.3f} "
          f"| lift {lg_sw:+.3f} (p_emp={pe_sw:.2f})  OK"); ok += 1

    print(f"\nALL {ok}/6 CORE TESTS PASSED")


if __name__ == "__main__":
    run()
