# Excerpt from the private inference service (inference.py, InferenceService).
# LIME's local fit is checked before its explanation is shown. This is the
# mechanism that flags an explanation as unreliable. Not runnable on its own:
# `result` comes from the LIME explainer and `probability` from the model.

from typing import Any


def add_explanation_quality(result: dict[str, Any], probability: float) -> None:
    # Compare the local surrogate's prediction with the model's own probability.
    error = abs(result["local_surrogate_prediction"] - probability)
    result["surrogate_absolute_error"] = float(error)

    # Flag the explanation when the surrogate fits poorly (R² < 0.8) or
    # disagrees with the model by more than 0.1. These are screening thresholds.
    if result["local_fidelity_r2"] < 0.8 or error > 0.1:
        result["quality_warning"] = (
            "The local approximation may be unreliable: R² is below 0.8 or the "
            "surrogate differs from the model probability by more than 0.1."
        )
    else:
        result["quality_warning"] = None
