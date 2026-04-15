#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageOps

from src.dataset import (
    D4_CANONICAL_SEQUENCES,
    PAD_TOKEN_ID,
    TransformTokenizer,
)
from src.model import ImageTransformPredictor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config_d4.yaml"

IDENTITY_MATRIX = ((1, 0), (0, 1))
ROTATE_MATRIX = ((0, -1), (1, 0))
REFLECT_MATRIX = ((-1, 0), (0, 1))


def _resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


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
    for element_name, tokens in D4_CANONICAL_SEQUENCES.items():
        matrix = IDENTITY_MATRIX
        for token in tokens:
            matrix = _apply_generator(matrix, token)
        lookup[matrix] = (element_name, list(tokens))
    return lookup


CANONICAL_LOOKUP = _build_canonical_lookup()


def canonicalize_tokens(tokens: List[str]) -> Tuple[str, List[str]]:
    matrix = IDENTITY_MATRIX
    for token in tokens:
        if token not in {"e", "r", "s"}:
            raise ValueError(f"Unsupported target token: {token}")
        matrix = _apply_generator(matrix, token)
    return CANONICAL_LOOKUP[matrix]


def normalize_target_spec(target_spec: str) -> Tuple[str, List[str]]:
    raw = target_spec.strip()
    compact = raw.replace(" ", "")
    aliases = {
        "r^2": "r2",
        "r^3": "r3",
        "sr^2": "sr2",
        "sr^3": "sr3",
    }
    compact = aliases.get(compact, compact)

    if compact in D4_CANONICAL_SEQUENCES:
        return compact, list(D4_CANONICAL_SEQUENCES[compact])

    tokens = raw.replace(",", " ").split()
    if not tokens:
        raise ValueError("Target must not be empty.")
    canonical_name, canonical_tokens = canonicalize_tokens(tokens)
    return canonical_name, canonical_tokens


def choose_device(device_arg: Optional[str], config) -> torch.device:
    requested = device_arg or config.training.get("device", "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA device is not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_pil_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def pil_to_tensor(image: Image.Image, preprocess, device: torch.device) -> torch.Tensor:
    return preprocess(image).unsqueeze(0).to(device)


def apply_token_to_pil(image: Image.Image, token: str) -> Image.Image:
    if token == "e":
        return image
    if token == "r":
        return image.rotate(90, expand=True)
    if token == "s":
        return ImageOps.mirror(image)
    raise ValueError(f"Unsupported D4 token for image transformation: {token}")


def apply_sequence_to_pil(image: Image.Image, tokens: List[str]) -> Image.Image:
    transformed = image.copy()
    for token in tokens:
        transformed = apply_token_to_pil(transformed, token)
    return transformed.convert("RGB")


def build_all_d4_candidates(base_image: Image.Image) -> List[Tuple[str, List[str], Image.Image]]:
    candidates: List[Tuple[str, List[str], Image.Image]] = []
    for element_name, tokens in D4_CANONICAL_SEQUENCES.items():
        transformed_image = apply_sequence_to_pil(base_image, tokens)
        candidates.append((element_name, list(tokens), transformed_image))
    return candidates


def load_model_checkpoint(
    model: ImageTransformPredictor,
    checkpoint_path: str,
    device: torch.device,
    config_path: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        encoder_type = model.config.encoder.get("type", "unknown")
        raise RuntimeError(
            "Failed to load checkpoint into the model. "
            f"Most likely the checkpoint/config pair is incompatible.\n"
            f"config: {config_path}\n"
            f"checkpoint: {checkpoint_path}\n"
            f"config encoder.type: {encoder_type}\n"
            "Use train_config_d4.yaml with d4_efficientnet checkpoints and "
            "train_config_d4_vit.yaml with d4_vit checkpoints."
        ) from exc


@torch.no_grad()
def score_target_sequence(
    model: ImageTransformPredictor,
    image1: torch.Tensor,
    image2: torch.Tensor,
    input_ids: torch.Tensor,
) -> Dict[str, float]:
    images_embeddings = model.image_pair_encoder(image1, image2)
    logits, _ = model.transform_decoder(
        idx=input_ids,
        images_embeddings=images_embeddings,
        targets=None,
    )
    targets = torch.full_like(input_ids, PAD_TOKEN_ID)
    targets[:, :-1] = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    mask = targets != PAD_TOKEN_ID

    sum_logprob = float(gathered[mask].sum().item())
    token_count = int(mask.sum().item())
    mean_logprob = sum_logprob / token_count if token_count > 0 else float("-inf")

    return {
        "sum_logprob": sum_logprob,
        "mean_logprob": mean_logprob,
        "token_count": token_count,
    }


def default_output_path(image_path: str, target_name: str) -> str:
    image = Path(image_path)
    return str(image.with_name(f"{image.stem}__model_{target_name}.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the D4-transformed image that best matches a requested target token sequence "
            "according to the checkpoint weights."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the training config YAML. For EfficientNet checkpoints use train_config_d4.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint (.pth).",
    )
    parser.add_argument(
        "--image1",
        type=str,
        required=True,
        help="Path to the source image.",
    )
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
        help="Where to save the selected output image. Default: <image_stem>__model_<target>.png",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        choices=("sum", "mean"),
        default="sum",
        help="How to rank candidate images: by total log-probability or mean log-probability per target token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    image1_path = _resolve_project_path(args.image1)

    config = OmegaConf.load(config_path)
    device = choose_device(args.device, config)

    target_name, target_tokens = normalize_target_spec(args.target)

    model = ImageTransformPredictor(config.model)
    model.to(device)
    model.eval()
    load_model_checkpoint(model, checkpoint_path, device, config_path)

    tokenizer = TransformTokenizer()
    preprocess = model.image_pair_encoder.preprocess
    input_ids = tokenizer.encode(
        transforms=target_tokens,
        add_special_tokens=True,
        max_seq_len=config.model.decoder.max_seq_len,
        return_targets=False,
    ).unsqueeze(0).to(device)

    base_image = load_pil_image(image1_path)
    image1_tensor = pil_to_tensor(base_image, preprocess, device)

    ranked_candidates = []
    for candidate_name, candidate_tokens, candidate_image in build_all_d4_candidates(base_image):
        image2_tensor = pil_to_tensor(candidate_image, preprocess, device)
        scores = score_target_sequence(
            model=model,
            image1=image1_tensor,
            image2=image2_tensor,
            input_ids=input_ids,
        )
        ranked_candidates.append(
            {
                "candidate_name": candidate_name,
                "candidate_tokens": candidate_tokens,
                "candidate_image": candidate_image,
                **scores,
            }
        )

    metric_name = "sum_logprob" if args.score_mode == "sum" else "mean_logprob"
    ranked_candidates.sort(key=lambda item: item[metric_name], reverse=True)
    best = ranked_candidates[0]

    output_path = _resolve_project_path(args.output) if args.output else default_output_path(image1_path, target_name)
    best["candidate_image"].save(output_path)

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"image1: {image1_path}")
    print(f"encoder: {config.model.encoder.type}")
    print(f"requested target: {args.target}")
    print(f"canonical target: {target_name}")
    print(f"canonical target tokens: {target_tokens}")
    print(f"score mode: {args.score_mode}")
    print("")
    print("candidate ranking:")
    for item in ranked_candidates:
        print(
            f"  {item['candidate_name']:>3} | tokens={item['candidate_tokens']} | "
            f"sum_logprob={item['sum_logprob']:.4f} | mean_logprob={item['mean_logprob']:.4f}"
        )
    print("")
    print(f"selected candidate: {best['candidate_name']}")
    print(f"selected candidate tokens: {best['candidate_tokens']}")
    print(f"saved output image: {output_path}")


if __name__ == "__main__":
    main()
