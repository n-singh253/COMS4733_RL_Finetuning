"""Vision-Language-Action model built from DINOv2 and BERT backbones."""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class VLADinoV2Config:
    vision_encoder: str = "facebook/dinov2-base"
    language_encoder: str = "bert-base-uncased"
    fusion_hidden_dim: int = 512
    fusion_layers: int = 2
    action_dim: int = 7
    max_instruction_tokens: int = 64
    freeze_vision: bool = True
    freeze_language: bool = False
    dropout: float = 0.1


class VLADinoV2Policy(nn.Module):
    """Minimal VLA policy combining pretrained vision/language encoders."""

    def __init__(self, config: VLADinoV2Config) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = AutoModel.from_pretrained(config.vision_encoder)
        self.language_encoder = AutoModel.from_pretrained(config.language_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_encoder)

        vision_dim = self.vision_encoder.config.hidden_size
        language_dim = self.language_encoder.config.hidden_size

        self.vision_projection = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, config.fusion_hidden_dim),
        )
        self.language_projection = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, config.fusion_hidden_dim),
        )
        self.proprio_projection = nn.Sequential(
            nn.Linear(7, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_hidden_dim,
            nhead=8,
            dim_feedforward=config.fusion_hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=config.fusion_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.Linear(config.fusion_hidden_dim, config.action_dim),
        )

        if config.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if config.freeze_language:
            for param in self.language_encoder.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb_static: torch.Tensor,
        instruction: List[str],
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Predict joint velocities from multimodal inputs.

        Args:
            rgb_static: Batch of RGB images ``(B, C, H, W)`` normalized to ``[0, 1]``.
            instruction: Batch of natural-language strings.
            proprio: Batch of proprioceptive vectors ``(B, 7)``.
        """

        device = rgb_static.device
        image_embedding = self._encode_image(rgb_static)
        text_embedding = self._encode_text(instruction, device=device)
        proprio_embedding = self.proprio_projection(proprio)

        fusion_tokens = torch.stack([image_embedding, text_embedding, proprio_embedding], dim=1)
        fused = self.fusion(fusion_tokens)
        pooled = fused.mean(dim=1)
        return self.head(pooled)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("Images must have shape (B, C, H, W)")

        outputs = self.vision_encoder(pixel_values=images)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.vision_projection(features)

    def _encode_text(self, instructions: Iterable[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.config.max_instruction_tokens,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.language_encoder(**tokens)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.language_projection(features)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "model_state": self.state_dict(),
            "config": dataclass_to_dict(self.config),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> "VLADinoV2Policy":
        config = dict_to_dataclass(VLADinoV2Config, checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state"])
        return model


def dataclass_to_dict(config: VLADinoV2Config) -> Dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


def dict_to_dataclass(cls, values: Dict[str, Any]):
    return cls(**values)


__all__ = ["VLADinoV2Config", "VLADinoV2Policy"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilities for the VLADinoV2Policy module")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a self-test forward pass with random dummy inputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size to use when executing the self-test.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image resolution to use for the dummy forward pass.",
    )
    return parser.parse_args()


def _run_self_test(batch_size: int, image_size: int) -> None:
    config = VLADinoV2Config()
    model = VLADinoV2Policy(config)
    model.eval()

    with torch.no_grad():
        dummy_rgb = torch.randn(batch_size, 3, image_size, image_size)
        dummy_proprio = torch.randn(batch_size, 7)
        dummy_instructions = ["test instruction"] * batch_size
        outputs = model(rgb_static=dummy_rgb, proprio=dummy_proprio, instruction=dummy_instructions)

    print("Self-test completed. Output shape:", tuple(outputs.shape))


def main() -> None:
    args = _parse_args()
    if args.test:
        _run_self_test(args.batch_size, args.image_size)
    else:
        print("No action specified. Use --test to run the built-in self-test.")


if __name__ == "__main__":  # pragma: no cover - CLI convenience wrapper
    main()
