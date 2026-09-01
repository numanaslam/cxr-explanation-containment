"""Containment metric for explanation grounding (model-independent core).

Precision, recall, IoU and lift-over-chance for a binary explanation mask A
against a binary reference (lung / lesion) mask B. Pure numpy -- no model, no
framework -- so it is identical whether explanations come from MATLAB, ONNX or
a native-Python pipeline, and it can be unit-tested with synthetic masks.

Data links: see ../code/datasets_download.m
Code:       https://github.com/numanaslam/cxr-explanation-containment
"""
from __future__ import annotations
import numpy as np


def _b(m) -> np.ndarray:
    return np.asarray(m).astype(bool)


def precision(A, B) -> float:
    """|A ∩ B| / |A|  -- fraction of the explanation inside the reference."""
    A, B = _b(A), _b(B)
    a = A.sum()
    return float((A & B).sum() / a) if a else float("nan")


def recall(A, B) -> float:
    """|A ∩ B| / |B|  -- reference coverage."""
    A, B = _b(A), _b(B)
    b = B.sum()
    return float((A & B).sum() / b) if b else float("nan")


def iou(A, B) -> float:
    A, B = _b(A), _b(B)
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else float("nan")


def reference_fraction(B) -> float:
    """Per-image chance level: reference area / frame area."""
    B = _b(B)
    return float(B.sum() / B.size)


def lift_analytic(A, B) -> float:
    """The paper's published lift: precision minus the reference's frame share.

    Assumes A is placed uniformly at random -- the baseline JMI review Concern
    (centrality) targets. Compare against lift_geometry() from nulls.py.
    """
    return precision(A, B) - reference_fraction(B)
