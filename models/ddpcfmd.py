"""Dual-Driven Physically Consistent Fusion for Multimodal Dehazing."""

from typing import NamedTuple

from torch import Tensor, nn

from .radiance import RadianceReconstruction
from .transmission import TransmissionEstimation


class DDPCFMDOutput(NamedTuple):
    """Polarization, transmission, airlight, and radiance estimates."""

    airlight_dop: Tensor
    transmission_dop: Tensor
    transmission: Tensor
    tgi_transmission: Tensor
    airlight: Tensor
    radiance: Tensor


class DDPCFMD(nn.Module):
    """Multimodal dehazing with aligned floating-point NCHW inputs.

    Input channels: VLP 9, intensity 3, polarization difference 3, LWIR 1, TGI 1.
    Values use the [0, 1] scale. Each spatial dimension must be at least 80
    and divisible by 8; image_size is fixed at construction.
    """

    def __init__(self, image_size=(240, 320)):
        super().__init__()
        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or any(type(size) is not int or size < 80 or size % 8 for size in image_size)
        ):
            raise ValueError("image_size must contain two multiples of 8, each at least 80.")
        self.image_size = tuple(image_size)
        self.transmission_estimation = TransmissionEstimation(self.image_size)
        self.radiance_reconstruction = RadianceReconstruction(self.image_size)

    def validate_modalities(self, vlp, intensity, polarization_difference, lwir, tgi):
        """Check input shapes, batch size, device, and dtype."""
        inputs = (
            ("vlp", vlp, 9),
            ("intensity", intensity, 3),
            ("polarization_difference", polarization_difference, 3),
            ("lwir", lwir, 1),
            ("tgi", tgi, 1),
        )
        for name, tensor, channels in inputs:
            if not isinstance(tensor, Tensor) or tensor.ndim != 4:
                raise ValueError(f"{name} must be a four-dimensional NCHW tensor.")
            if tensor.shape[1] != channels or tuple(tensor.shape[2:]) != self.image_size:
                raise ValueError(
                    f"{name} must have {channels} channels and spatial size {self.image_size}."
                )
            if not tensor.is_floating_point() or tensor.shape[0] < 1:
                raise ValueError(f"{name} must be floating point with a nonempty batch.")
            if (
                tensor.shape[0] != vlp.shape[0]
                or tensor.device != vlp.device
                or tensor.dtype != vlp.dtype
            ):
                raise ValueError("All modalities must share batch size, device and dtype.")

    def forward(self, vlp, intensity, polarization_difference, lwir, tgi):
        self.validate_modalities(vlp, intensity, polarization_difference, lwir, tgi)
        airlight_dop, transmission_dop, tgi_transmission, transmission = (
            self.transmission_estimation(vlp, intensity, polarization_difference, tgi)
        )
        airlight, radiance = self.radiance_reconstruction(vlp, intensity, transmission, lwir)
        return DDPCFMDOutput(
            airlight_dop, transmission_dop, transmission, tgi_transmission, airlight, radiance
        )
