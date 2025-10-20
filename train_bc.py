"""Behavioral cloning training entry-point for the VLA model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from data.lerobot_dataset import LeRobotDataset
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from utils.config import load_config, save_config
from utils.logging import MetricTracker, get_logger, setup_logging
from utils.seed import seed_everything


def build_dataloader(
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    split: str,
) -> Tuple[LeRobotDataset, DataLoader]:
    dataset_root = dataset_cfg["path"]
    sequence_length = dataset_cfg.get("sequence_length", 1)
    normalize_proprio = dataset_cfg.get("normalize_proprio", True)

    encoder_name = model_cfg.get("vision_encoder", "facebook/dinov2-base")
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)

    def image_transform(image):
        inputs = image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    dataset = LeRobotDataset(
        root=dataset_root,
        split=split,
        sequence_length=sequence_length,
        image_transform=image_transform,
        normalize_proprio=normalize_proprio,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dataset_cfg["batch_size"],
        shuffle=split == dataset_cfg.get("train_split", "train"),
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    return dataset, dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DINOv2+BERT VLA model via behavioral cloning.")
    parser.add_argument("--config", type=str, default="configs/openvla_dinov2_bc.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs declared in the YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load a single batch and execute a forward/backward pass without writing checkpoints. "
            "Useful for smoke-testing the setup."
        ),
    )
    args = parser.parse_args()

    raw_config = load_config(args.config, overrides=args.override)
    dataset_cfg: Dict[str, Any] = dict(raw_config.get("dataset", {}))
    model_cfg: Dict[str, Any] = dict(raw_config.get("model", {}))
    training_cfg: Dict[str, Any] = dict(raw_config.get("training", {}))

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs

    setup_logging()
    logger = get_logger("train_bc")

    seed_everything(training_cfg.get("seed", 0), deterministic_cudnn=training_cfg.get("deterministic", False))

    device = torch.device(training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    _, train_loader = build_dataloader(dataset_cfg, model_cfg, dataset_cfg.get("train_split", "train"))
    _, val_loader = build_dataloader(dataset_cfg, model_cfg, dataset_cfg.get("val_split", "val"))

    model_config = VLADinoV2Config(**model_cfg)
    model = VLADinoV2Policy(model_config).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["epochs"])
        if training_cfg.get("scheduler", "cosine") == "cosine"
        else None
    )
    criterion = nn.MSELoss()

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "./runs"))
    if not args.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_config(raw_config, checkpoint_dir / f"{training_cfg.get('checkpoint_name', 'bc_run')}_config.yaml")

    if args.dry_run:
        logger.info("Running dry run with a single batch from the training loader.")
        batch = next(iter(train_loader))
        model.train()
        optimizer.zero_grad()
        rgb = batch["rgb_static"].to(device)
        proprio = batch["proprio"].to(device)
        target = batch["action"].to(device)
        instructions = batch.get("instruction") or [""] * rgb.size(0)
        outputs = model(rgb_static=rgb, proprio=proprio, instruction=instructions)
        loss = criterion(outputs, target)
        loss.backward()
        grad_clip = training_cfg.get("grad_clip", 0.0)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        logger.info("Dry run completed successfully (loss=%.5f).", float(loss.item()))
        return

    best_val_loss = float("inf")

    for epoch in range(1, training_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training_cfg.get("log_interval", 10),
            training_cfg.get("grad_clip", 0.0),
        )
        logger.info("Epoch %d | train_loss=%.5f", epoch, train_loss)

        if scheduler is not None:
            scheduler.step()

        if (epoch % training_cfg.get("val_interval", 1)) == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            logger.info("Epoch %d | val_loss=%.5f", epoch, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / f"{training_cfg.get('checkpoint_name', 'bc_run')}_best.pt"
                logger.info("New best checkpoint at %s", checkpoint_path)
                torch.save({"model_state": model.state_dict(), "config": model_config.__dict__}, checkpoint_path)

        last_checkpoint_path = checkpoint_dir / f"{training_cfg.get('checkpoint_name', 'bc_run')}_last.pt"
        torch.save({"model_state": model.state_dict(), "config": model_config.__dict__}, last_checkpoint_path)


def train_one_epoch(
    model: VLADinoV2Policy,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
    grad_clip: float,
) -> float:
    model.train()
    tracker = MetricTracker()

    for step, batch in enumerate(dataloader, start=1):
        rgb = batch["rgb_static"].to(device)
        proprio = batch["proprio"].to(device)
        target = batch["action"].to(device)
        instructions = batch.get("instruction")
        if instructions is None:
            instructions = [""] * rgb.size(0)

        optimizer.zero_grad()
        pred = model(rgb_static=rgb, proprio=proprio, instruction=instructions)
        loss = criterion(pred, target)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        tracker.update(loss.item(), rgb.size(0))

        if step % log_interval == 0:
            get_logger("train_bc").info("Step %d | loss=%.5f", step, tracker.average)

    return tracker.average


def evaluate(model: VLADinoV2Policy, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    tracker = MetricTracker()
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb_static"].to(device)
            proprio = batch["proprio"].to(device)
            target = batch["action"].to(device)
            instructions = batch.get("instruction")
            if instructions is None:
                instructions = [""] * rgb.size(0)

            pred = model(rgb_static=rgb, proprio=proprio, instruction=instructions)
            loss = criterion(pred, target)
            tracker.update(loss.item(), rgb.size(0))
    return tracker.average


if __name__ == "__main__":
    main()
