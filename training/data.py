"""Multimodal training samples in NPY/PNG format."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

STAGE_FIELDS = {
    "transmission": (
        "vlp",
        "intensity",
        "polarization_difference",
        "airlight_dop",
        "transmission_dop",
        "transmission",
        "tgi",
    ),
    "radiance": ("vlp", "intensity", "transmission", "airlight", "radiance", "lwir"),
    "joint": (
        "vlp",
        "intensity",
        "polarization_difference",
        "airlight_dop",
        "transmission_dop",
        "transmission",
        "tgi",
        "airlight",
        "radiance",
        "lwir",
    ),
}

# Map physical quantities to the NPY/PNG dataset folders.
FIELD_FILES = {
    "vlp": ("I_alpha", ".npy", 9),
    "intensity": ("I_hat", ".png", 3),
    "polarization_difference": ("delta_I_hat", ".png", 3),
    "airlight_dop": ("P_A", ".npy", 3),
    "transmission_dop": ("P_T", ".png", 3),
    "transmission": ("T", ".png", 3),
    "tgi": ("gated", ".png", 1),
    "airlight": ("A_infinity", ".png", 3),
    "radiance": ("R", ".png", 3),
    "lwir": ("ir_foggy", ".png", 1),
}


class MultimodalTrainingDataset(Dataset):
    """Load stage-specific training fields and resize them with bilinear interpolation."""

    def __init__(self, data_root, stage, image_size=(240, 320)):
        if stage not in STAGE_FIELDS:
            raise ValueError(f"Unknown training stage: {stage}")
        self.data_root = Path(data_root)
        self.stage = stage
        self.image_size = tuple(image_size)
        self.fields = STAGE_FIELDS[stage]
        for field in self.fields:
            folder = self.data_root / FIELD_FILES[field][0]
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing training directory for {field}: {folder}")
        self.sample_names = sorted(path.stem for path in (self.data_root / "I_alpha").glob("*.npy"))
        if not self.sample_names:
            raise ValueError("The I_alpha directory contains no training samples.")

    def __len__(self):
        return len(self.sample_names)

    def load_training_field(self, field, sample_name):
        folder, extension, channels = FIELD_FILES[field]
        path = self.data_root / folder / (sample_name + extension)
        if extension == ".npy":
            array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            if field == "airlight_dop":
                if array.shape != (3,):
                    raise ValueError(f"{path} must contain a three-element airlight DoP vector.")
                tensor = torch.from_numpy(array.copy())[:, None, None]
                tensor = tensor.expand(-1, *self.image_size).contiguous()
            else:
                if array.ndim != 3 or array.shape[-1] != channels:
                    raise ValueError(f"{path} must have H x W x {channels} layout.")
                tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        else:
            with Image.open(path) as image:
                array = (
                    np.array(image.convert("L" if channels == 1 else "RGB"), dtype=np.float32)
                    / 255.0
                )
            array = array[None, ...] if channels == 1 else array.transpose(2, 0, 1)
            tensor = torch.from_numpy(np.ascontiguousarray(array))
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values in training sample: {path}")
        if tuple(tensor.shape[1:]) != self.image_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        return tensor

    def __getitem__(self, index):
        sample_name = self.sample_names[index]
        return {
            **{field: self.load_training_field(field, sample_name) for field in self.fields},
            "name": sample_name,
        }
