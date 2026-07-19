"""Inactive legacy comparator retained for reproducibility.

The implementation used a fold-0 global covariance selection whose participant
group later served as outer test. It is not the active Hormonbench v1 reference.
"""

from .model import (
    CustomParameters,
    JointBayesPersonalizer,
    estimate_custom_parameters,
    learn_conformal_multipliers,
    learn_lambdas,
    posterior_update,
)

__all__ = [
    "CustomParameters",
    "JointBayesPersonalizer",
    "estimate_custom_parameters",
    "learn_conformal_multipliers",
    "learn_lambdas",
    "posterior_update",
]
