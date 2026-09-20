"""Spatial normalization layers."""

import functools
from torch import nn


def spatial_normalization(norm_type="instance"):
    """Select instance or batch normalization."""
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d)
    if norm_type == "batch":
        return nn.BatchNorm2d
    raise ValueError(f"Unsupported spatial normalization: {norm_type}")
