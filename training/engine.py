"""Stage-wise training and checkpoint handling."""

import math
import os
import tempfile
from functools import partial
from pathlib import Path

import torch
from torch import nn

from models import DDPCFMD
from models.radiance import RadianceReconstruction
from models.transmission import TransmissionEstimation
from .losses import PhysicallyConsistentReconstructionLoss


def initialize_training_weights(module):
    """Apply the research training initialization before loading checkpoints."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_training_model(stage, image_size, device="cpu"):
    architectures = {
        "transmission": TransmissionEstimation,
        "radiance": RadianceReconstruction,
        "joint": DDPCFMD,
    }
    model = architectures[stage](image_size=image_size).to(device)
    model.apply(initialize_training_weights)
    return model


def learning_rate_multiplier(epoch, schedule):
    if schedule == "branch_decay":
        # MultiplicativeLR applies this factor to the current learning rate.
        return -epoch / 200 + 2.5 if epoch > 300 else 1
    if schedule == "constant":
        return 1
    raise ValueError(f"Unknown learning-rate schedule: {schedule}")


def training_predictions(model, stage, sample):
    """Route stage inputs and return predictions by physical quantity."""
    if stage == "transmission":
        airlight_dop, transmission_dop, tgi_transmission, transmission = model(
            sample["vlp"],
            sample["intensity"],
            sample["polarization_difference"],
            sample["tgi"],
        )
        return dict(
            airlight_dop=airlight_dop,
            transmission_dop=transmission_dop,
            transmission=transmission,
            tgi_transmission=tgi_transmission,
        )
    if stage == "radiance":
        airlight, radiance = model(
            sample["vlp"], sample["intensity"], sample["transmission"], sample["lwir"]
        )
        return dict(airlight=airlight, radiance=radiance)
    return model(
        sample["vlp"],
        sample["intensity"],
        sample["polarization_difference"],
        sample["lwir"],
        sample["tgi"],
    )._asdict()


def load_training_checkpoint(path, expected_stage, image_size):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != "ddpcfmd-training-v1":
        raise ValueError("Use a checkpoint created by this public training code.")
    if state.get("stage") != expected_stage or tuple(state.get("image_size", ())) != tuple(
        image_size
    ):
        raise ValueError(
            "Checkpoint stage or input size does not match the requested model."
        )
    return state


def initialize_joint_training(
    model, image_size, transmission_path=None, radiance_path=None
):
    for stage, path, branch in (
        ("transmission", transmission_path, model.transmission_estimation),
        ("radiance", radiance_path, model.radiance_reconstruction),
    ):
        if path is not None:
            state = load_training_checkpoint(path, stage, image_size)
            branch.load_state_dict(state["model"], strict=True)


class DDPCFMDTrainer:
    """Optimize one training stage and save its latest checkpoint."""

    def __init__(self, model, stage, data_loader, config, device, output_dir, resume=None):
        check_training_config(config, stage)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.stage = stage
        self.data_loader = data_loader
        self.config = config
        self.loss = PhysicallyConsistentReconstructionLoss(stage, config["loss_weights"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), **config["optimizer"])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lr_lambda=partial(learning_rate_multiplier, schedule=config["lr_schedule"]),
        )
        self.training_options = {
            "optimizer": config["optimizer"],
            "loss_weights": config["loss_weights"],
            "lr_schedule": config["lr_schedule"],
        }
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / f"{stage}-last.pth"
        self.start_epoch = 1
        if self.checkpoint_path.exists() and (
            resume is None or Path(resume).resolve() != self.checkpoint_path.resolve()
        ):
            raise FileExistsError(
                "A training checkpoint already exists here; use --resume or a new output directory."
            )
        if resume is not None:
            self.resume_training(resume)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resume_training(self, path):
        state = load_training_checkpoint(path, self.stage, self.config["image_size"])
        if state["training_options"] != self.training_options:
            raise ValueError(
                "Resume with the saved optimizer/loss/scheduler settings; epochs may be increased."
            )
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.start_epoch = state["epoch"] + 1
        if self.start_epoch > self.config["epochs"]:
            raise ValueError(
                "The checkpoint has already reached the configured total epochs."
            )

    def train_epoch(self, epoch):
        self.model.train()
        loss_sum, sample_count = 0.0, 0
        for sample in self.data_loader:
            sample = {
                name: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for name, value in sample.items()
            }
            self.optimizer.zero_grad(set_to_none=True)
            predictions = training_predictions(self.model, self.stage, sample)
            loss = self.loss(predictions, sample)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            self.optimizer.step()
            batch_size = sample["vlp"].shape[0]
            loss_sum += loss.detach().item() * batch_size
            sample_count += batch_size
        if not sample_count:
            raise ValueError("No training samples were loaded.")
        self.lr_scheduler.step()
        return loss_sum / sample_count

    def save_training_checkpoint(self, epoch):
        state = {
            "format": "ddpcfmd-training-v1",
            "stage": self.stage,
            "image_size": list(self.config["image_size"]),
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "training_options": self.training_options,
        }
        descriptor, temporary = tempfile.mkstemp(
            dir=self.output_dir, prefix="checkpoint-", suffix=".tmp"
        )
        os.close(descriptor)
        try:
            torch.save(state, temporary)
            os.replace(temporary, self.checkpoint_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def train(self):
        final_loss = None
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            final_loss = self.train_epoch(epoch)
            print(
                f"{self.stage} | epoch {epoch}/{self.config['epochs']} | training loss {final_loss:.6f}"
            )
            if epoch % self.config["save_every"] == 0 or epoch == self.config["epochs"]:
                self.save_training_checkpoint(epoch)
        return final_loss


def check_training_config(config, stage=None):
    required = {
        "stage",
        "image_size",
        "batch_size",
        "epochs",
        "num_workers",
        "shuffle",
        "optimizer",
        "lr_schedule",
        "loss_weights",
        "save_every",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(
            "Use the fields in configs/<stage>.json; unknown or missing fields are not accepted."
        )
    if config["stage"] not in ("transmission", "radiance", "joint"):
        raise ValueError("Unknown training stage in configuration.")
    if stage is not None and config["stage"] != stage:
        raise ValueError("Configuration stage does not match --stage.")
    if type(config["shuffle"]) is not bool:
        raise ValueError("shuffle must be a boolean.")
    size = config["image_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(x) is not int or x < 80 or x % 8 for x in size)
    ):
        raise ValueError("image_size must contain two multiples of 8, each at least 80.")
    for name in ("batch_size", "epochs", "save_every", "num_workers"):
        minimum = 0 if name == "num_workers" else 1
        if type(config[name]) is not int or config[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if learning_rate_multiplier(config["epochs"], config["lr_schedule"]) <= 0:
        raise ValueError(
            "branch_decay requires epochs < 500; choose constant for longer runs."
        )
    optimizer = config["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {
        "lr",
        "betas",
        "weight_decay",
        "amsgrad",
    }:
        raise ValueError("Specify optimizer lr, betas, weight_decay and amsgrad.")
    for name, value, strictly_positive in (
        ("lr", optimizer["lr"], True),
        ("weight_decay", optimizer["weight_decay"], False),
    ):
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or value < 0
            or (strictly_positive and value == 0)
        ):
            raise ValueError(f"Invalid {name}.")
    betas = optimizer["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(type(x) not in (int, float) or not 0 <= x < 1 for x in betas)
    ):
        raise ValueError("Adam betas must contain two values in [0, 1).")
    if type(optimizer["amsgrad"]) is not bool or not isinstance(
        config["loss_weights"], dict
    ):
        raise ValueError("Invalid amsgrad or loss_weights value.")
    PhysicallyConsistentReconstructionLoss(config["stage"], config["loss_weights"])
