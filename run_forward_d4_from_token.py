#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent

IDENTITY_MATRIX = ((1, 0), (0, 1))
ROTATE_MATRIX = ((0, -1), (1, 0))
REFLECT_MATRIX = ((-1, 0), (0, 1))

CANONICAL_D4_SEQUENCES: Dict[str, List[str]] = {
    "e": ["e"],
    "r": ["r"],
    "r2": ["r", "r"],
    "r3": ["r", "r", "r"],
    "s": ["s"],
    "sr": ["s", "r"],
    "sr2": ["s", "r", "r"],
    "sr3": ["s", "r", "r", "r"],
}


def _matmul(
    a: Tuple[Tuple[int, int], Tuple[int, int]],
    b: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ),
    )


def _apply_generator(
    current: Tuple[Tuple[int, int], Tuple[int, int]],
    token: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if token == "e":
        return current
    if token == "r":
        return _matmul(ROTATE_MATRIX, current)
    if token == "s":
        return _matmul(REFLECT_MATRIX, current)
    raise ValueError(f"Unsupported D4 token: {token}")


def _build_canonical_lookup() -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[str, List[str]]]:
    lookup: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[str, List[str]]] = {}
    for element_name, tokens in CANONICAL_D4_SEQUENCES.items():
        matrix = IDENTITY_MATRIX
        for token in tokens:
            matrix = _apply_generator(matrix, token)
        lookup[matrix] = (element_name, list(tokens))
    return lookup


CANONICAL_LOOKUP = _build_canonical_lookup()


def normalize_target_spec(target_spec: str) -> Tuple[str, List[str], Tuple[Tuple[int, int], Tuple[int, int]]]:
    raw = target_spec.strip()
    compact = raw.replace(" ", "")
    aliases = {
        "r^2": "r2",
        "r^3": "r3",
        "sr^2": "sr2",
        "sr^3": "sr3",
    }
    compact = aliases.get(compact, compact)

    if compact in CANONICAL_D4_SEQUENCES:
        tokens = list(CANONICAL_D4_SEQUENCES[compact])
    else:
        tokens = raw.replace(",", " ").split()
        if not tokens:
            raise ValueError("Target must not be empty.")

    matrix = IDENTITY_MATRIX
    for token in tokens:
        matrix = _apply_generator(matrix, token)

    canonical_name, canonical_tokens = CANONICAL_LOOKUP[matrix]
    return canonical_name, canonical_tokens, matrix


class TokenConditionedD4Warp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _needs_swap(matrix: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        return abs(matrix[0][1]) == 1 and abs(matrix[1][0]) == 1

    @staticmethod
    def _to_theta(
        matrix: Tuple[Tuple[int, int], Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(
            [[matrix[0][0], matrix[0][1], 0.0], [matrix[1][0], matrix[1][1], 0.0]],
            device=device,
            dtype=dtype,
        ).unsqueeze(0)

    def forward(
        self,
        image: torch.Tensor,
        tokens: List[str],
    ) -> torch.Tensor:
        if image.dim() != 4 or image.size(0) != 1:
            raise ValueError("Expected image tensor of shape [1, C, H, W].")

        matrix = IDENTITY_MATRIX
        for token in tokens:
            matrix = _apply_generator(matrix, token)

        _, channels, height, width = image.shape
        if self._needs_swap(matrix):
            out_height, out_width = width, height
        else:
            out_height, out_width = height, width

        theta = self._to_theta(matrix, image.device, image.dtype)
        grid = F.affine_grid(
            theta,
            size=(1, channels, out_height, out_width),
            align_corners=False,
        )
        warped = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped


def load_pil_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(image).unsqueeze(0)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    return transforms.ToPILImage()(image)


def default_output_path(image_path: str, target_name: str) -> str:
    image = Path(image_path)
    return str(image.with_name(f"{image.stem}__forward_{target_name}.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a D4 token sequence to an image via a pure PyTorch forward pass."
    )
    parser.add_argument("--image1", type=str, required=True, help="Path to the source image.")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help='Requested D4 target. Examples: "r", "r2", "r3", "s", "sr", "r r r", "r^3".',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the output image. Default: <image_stem>__forward_<target>.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    canonical_name, canonical_tokens, matrix = normalize_target_spec(args.target)
    image_path = str(Path(args.image1))
    output_path = args.output or default_output_path(image_path, canonical_name)

    image = load_pil_image(image_path)
    image_tensor = pil_to_tensor(image)

    warper = TokenConditionedD4Warp()
    output_tensor = warper(image_tensor, canonical_tokens)
    output_image = tensor_to_pil(output_tensor)
    output_image.save(output_path)

    print(f"image1: {image_path}")
    print(f"requested target: {args.target}")
    print(f"canonical target: {canonical_name}")
    print(f"canonical tokens: {canonical_tokens}")
    print(f"matrix: {matrix}")
    print(f"saved output image: {output_path}")


if __name__ == "__main__":
    main()
