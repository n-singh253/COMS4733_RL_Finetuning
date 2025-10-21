"""Dataset utilities for loading LeRobot-format demonstrations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeIndex:
    """Metadata describing a single LeRobot episode."""

    path: Path
    length: int
    instruction: str
    split: str = "train"


class LeRobotDataset(Dataset):
    """PyTorch dataset for LeRobot-formatted demonstrations.

    The dataset expects the following structure within ``root``::

        root/
          episode_0001/
            rgb_static/
              000000.png
              ...
            proprio.npy
            actions.npy
            instruction.txt
            meta.json (optional)

    A top-level ``metadata.json`` file can optionally define dataset splits by
    listing episode directory names under ``{"splits": {"train": [...], ...}}``.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        modalities: Optional[Sequence[str]] = None,
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        sequence_length: int = 1,
        normalize_proprio: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if modalities is None:
            modalities = ("rgb_static", "proprio", "action", "instruction")
        self.modalities = tuple(modalities)
        self.image_transform = image_transform
        self.sequence_length = sequence_length
        self.normalize_proprio = normalize_proprio

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.episodes = self._discover_episodes(split)
        if not self.episodes:
            raise RuntimeError(f"No episodes found for split '{split}' in {self.root}")

        self.index: List[Tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            for t in range(episode.length):
                if t + self.sequence_length <= episode.length:
                    self.index.append((episode_idx, t))

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor | str]:
        episode_idx, start_t = self.index[item]
        episode = self.episodes[episode_idx]
        data = self._load_episode(episode)

        end_t = start_t + self.sequence_length

        sample: Dict[str, torch.Tensor | str] = {"episode_id": episode.path.name, "timestep": start_t}
        if "rgb_static" in self.modalities:
            frames = [self._load_image(path) for path in data["rgb_static"][start_t:end_t]]
            if self.sequence_length == 1:
                sample["rgb_static"] = frames[0]
            else:
                sample["rgb_static"] = torch.stack(frames, dim=0)

        if "proprio" in self.modalities:
            proprio = torch.from_numpy(data["proprio"][start_t:end_t]).float()
            if self.normalize_proprio:
                proprio = _normalize_range(proprio, -1.0, 1.0)
            sample["proprio"] = proprio if self.sequence_length > 1 else proprio.squeeze(0)

        if "action" in self.modalities:
            action = torch.from_numpy(data["actions"][start_t:end_t]).float()
            sample["action"] = action if self.sequence_length > 1 else action.squeeze(0)

        if "instruction" in self.modalities:
            sample["instruction"] = episode.instruction

        return sample

    # ------------------------------------------------------------------
    # Collation helpers
    # ------------------------------------------------------------------
    def collate_fn(self, batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor]:
        batch_tensors: Dict[str, List[torch.Tensor]] = {}
        instructions: List[str] = []
        meta: Dict[str, List[str | int]] = {"episode_id": [], "timestep": []}

        for item in batch:
            meta["episode_id"].append(item["episode_id"])
            meta["timestep"].append(int(item["timestep"]))

            if "rgb_static" in item:
                batch_tensors.setdefault("rgb_static", []).append(item["rgb_static"])  # type: ignore[arg-type]
            if "proprio" in item:
                batch_tensors.setdefault("proprio", []).append(item["proprio"])  # type: ignore[arg-type]
            if "action" in item:
                batch_tensors.setdefault("action", []).append(item["action"])  # type: ignore[arg-type]
            if "instruction" in item:
                instructions.append(item["instruction"])  # type: ignore[arg-type]

        collated: Dict[str, torch.Tensor | List[str] | List[str | int]] = {"meta": meta}
        for key, tensors in batch_tensors.items():
            collated[key] = torch.stack(tensors, dim=0)
        if instructions:
            collated["instruction"] = instructions
        return collated  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _discover_episodes(self, split: str) -> List[EpisodeIndex]:
        metadata_path = self.root / "metadata.json"
        split_map: Dict[str, List[str]] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            split_map = metadata.get("splits", {})

        candidates: Iterable[Path]
        if split_map and split in split_map:
            candidates = [self.root / episode for episode in split_map[split]]
        else:
            candidates = sorted(path for path in self.root.iterdir() if path.is_dir())

        episodes: List[EpisodeIndex] = []
        for episode_path in candidates:
            actions_path = episode_path / "actions.npy"
            instruction_path = episode_path / "instruction.txt"
            if not actions_path.exists() or not instruction_path.exists():
                continue
            actions = np.load(actions_path)
            length = int(actions.shape[0])
            with instruction_path.open("r", encoding="utf-8") as handle:
                instruction = handle.read().strip()
            episodes.append(EpisodeIndex(path=episode_path, length=length, instruction=instruction, split=split))
        return episodes

    def _load_episode(self, episode: EpisodeIndex) -> Dict[str, np.ndarray | List[Path]]:
        cache_key = episode.path
        if not hasattr(self, "_episode_cache"):
            self._episode_cache = {}
        if cache_key in self._episode_cache:
            return self._episode_cache[cache_key]

        # Images and proprio are in the "obs" subdirectory
        rgb_dir = episode.path / "obs" / "rgb_static"
        image_paths = []
        if rgb_dir.exists():
            image_paths = sorted(rgb_dir.glob("*.png"))
            if not image_paths:
                image_paths = sorted(rgb_dir.glob("*.jpg"))

        # Try obs/proprio.npy first (new format), fall back to proprio.npy (old format)
        proprio_path = episode.path / "obs" / "proprio.npy"
        if not proprio_path.exists():
            proprio_path = episode.path / "proprio.npy"
        
        actions_path = episode.path / "actions.npy"
        proprio = np.load(proprio_path) if proprio_path.exists() else np.zeros((episode.length, 7), dtype=np.float32)
        actions = np.load(actions_path)

        data = {"rgb_static": image_paths, "proprio": proprio, "actions": actions}
        self._episode_cache[cache_key] = data
        return data

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(image)
        array = torch.from_numpy(np.array(image)).float() / 255.0
        return array.permute(2, 0, 1)


def _normalize_range(tensor: torch.Tensor, low: float, high: float) -> torch.Tensor:
    minimum = tensor.min()
    maximum = tensor.max()
    if maximum == minimum:
        return torch.zeros_like(tensor)
    scaled = (tensor - minimum) / (maximum - minimum)
    return scaled * (high - low) + low


__all__ = ["LeRobotDataset", "EpisodeIndex"]
