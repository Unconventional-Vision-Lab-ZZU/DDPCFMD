"""Weighted L1/L2 supervision for physical predictions."""

import math

from torch import nn
from torch.nn import functional as F

STAGE_TARGETS = {
    "transmission": {
        "airlight_dop": "airlight_dop",
        "transmission_dop": "transmission_dop",
        "transmission": "transmission",
        "tgi_transmission": "transmission",
    },
    "radiance": {"airlight": "airlight", "radiance": "radiance"},
}
STAGE_TARGETS["joint"] = {**STAGE_TARGETS["transmission"], **STAGE_TARGETS["radiance"]}


class PhysicallyConsistentReconstructionLoss(nn.Module):
    """Weighted L1/L2 loss for each predicted physical quantity.

    Each weights entry is an [L1, L2] pair. Both transmission predictions
    share the transmitted-light target.
    """

    def __init__(self, stage, weights):
        super().__init__()
        if stage not in STAGE_TARGETS:
            raise ValueError(f"Unknown training stage: {stage}")
        self.targets = STAGE_TARGETS[stage]
        unknown = set(weights) - set(STAGE_TARGETS["joint"])
        if unknown:
            raise ValueError(f"Unknown reconstruction-loss terms: {sorted(unknown)}")
        self.weights = {}
        for prediction in self.targets:
            pair = weights.get(prediction)
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in pair
                )
                or sum(pair) <= 0
            ):
                raise ValueError(
                    f"Specify nonnegative [L1, L2] weights with a positive sum for {prediction}."
                )
            self.weights[prediction] = tuple(float(value) for value in pair)

    def forward(self, predictions, sample):
        terms = []
        for prediction, target in self.targets.items():
            estimate, reference = predictions[prediction], sample[target]
            if estimate.shape != reference.shape:
                raise ValueError(f"Prediction/target shape mismatch for {prediction}.")
            l1_weight, l2_weight = self.weights[prediction]
            terms.append(l1_weight * F.l1_loss(estimate, reference))
            terms.append(l2_weight * F.mse_loss(estimate, reference))
        return sum(terms)
