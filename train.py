"""DDPCFMD training entry point."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.data import MultimodalTrainingDataset
from training.engine import (
    DDPCFMDTrainer,
    build_training_model,
    check_training_config,
    initialize_joint_training,
)


def train_ddpcfmd():
    parser = argparse.ArgumentParser(description="DDPCFMD training")
    parser.add_argument(
        "--stage", choices=("transmission", "radiance", "joint"), required=True
    )
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Prepared training samples only."
    )
    parser.add_argument("--config", type=Path, help="Defaults to configs/<stage>.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--resume", type=Path, help="Continue a checkpoint from this public training code."
    )
    parser.add_argument(
        "--transmission-weights",
        type=Path,
        help="Initialize the joint model's transmission branch.",
    )
    parser.add_argument(
        "--radiance-weights",
        type=Path,
        help="Initialize the joint model's radiance branch.",
    )
    args = parser.parse_args()
    if (args.transmission_weights or args.radiance_weights) and (
        args.stage != "joint" or args.resume
    ):
        parser.error("Stage initialization weights are for a new joint run only.")
    config_path = args.config or Path(__file__).parent / "configs" / f"{args.stage}.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    check_training_config(config, args.stage)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu.")
    dataset = MultimodalTrainingDataset(args.data_root, args.stage, config["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = build_training_model(args.stage, config["image_size"], device)
    if args.stage == "joint":
        initialize_joint_training(
            model, config["image_size"], args.transmission_weights, args.radiance_weights
        )
    trainer = DDPCFMDTrainer(
        model, args.stage, loader, config, device, args.output_dir, args.resume
    )
    trainer.train()


if __name__ == "__main__":
    train_ddpcfmd()
