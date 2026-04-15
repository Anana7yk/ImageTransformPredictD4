#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from src.dataset import END_TOKEN_ID, PAD_TOKEN_ID, START_TOKEN_ID, TransformTokenizer
from src.model import ImageTransformPredictor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config_d4.yaml"

IDENTITY_MATRIX = ((1, 0), (0, 1))
ROTATE_MATRIX = ((0, -1), (1, 0))
REFLECT_MATRIX = ((-1, 0), (0, 1))

ROTATION_SEQUENCES: Dict[str, List[str]] = {
    "e": ["e"],
    "r": ["r"],
    "r2": ["r", "r"],
    "r3": ["r", "r", "r"],
}

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


def _resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _matmul(a: Tuple[Tuple[int, int], Tuple[int, int]], b: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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


def _apply_generator(current: Tuple[Tuple[int, int], Tuple[int, int]], token: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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


def canonicalize_prediction(tokens: List[str]) -> Dict[str, object]:
    if tokens == ["[NULL]"]:
        return {
            "raw_tokens": tokens,
            "decoded_chain": "∅",
            "canonical_element": "∅",
            "canonical_tokens": ["[NULL]"],
            "is_canonical": None,
            "is_null": True,
            "status": "unrelated",
        }

    if "[NULL]" in tokens:
        return {
            "raw_tokens": tokens,
            "decoded_chain": " ".join(tokens) if tokens else "(empty)",
            "canonical_element": "invalid",
            "canonical_tokens": None,
            "is_canonical": False,
            "is_null": True,
            "status": "invalid_mixed_null_sequence",
        }

    matrix = IDENTITY_MATRIX
    for token in tokens:
        if token not in {"e", "r", "s"}:
            return {
                "raw_tokens": tokens,
                "decoded_chain": " ".join(tokens) if tokens else "(empty)",
                "canonical_element": "invalid",
                "canonical_tokens": None,
                "is_canonical": False,
                "is_null": False,
                "status": "invalid_unknown_token",
            }
        matrix = _apply_generator(matrix, token)

    canonical_element, canonical_tokens = CANONICAL_LOOKUP[matrix]
    return {
        "raw_tokens": tokens,
        "decoded_chain": " ".join(tokens) if tokens else "(empty)",
        "canonical_element": canonical_element,
        "canonical_tokens": canonical_tokens,
        "is_canonical": tokens == canonical_tokens,
        "is_null": False,
        "status": "related",
    }


def load_model_checkpoint(model: ImageTransformPredictor, checkpoint_path: str, device: torch.device, config_path: str) -> None:
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


def apply_rotation_sequence(image: Image.Image, tokens: List[str]) -> Image.Image:
    transformed = image.copy()
    for token in tokens:
        if token == "e":
            continue
        if token == "r":
            transformed = transformed.rotate(90, expand=True)
            continue
        raise ValueError(f"Rotation probe received non-rotation token: {token}")
    return transformed.convert("RGB")


@torch.no_grad()
def autoregressive_decode(
    model: ImageTransformPredictor,
    image1: torch.Tensor,
    image2: torch.Tensor,
    tokenizer: TransformTokenizer,
    max_gen_len: int,
) -> Dict[str, object]:
    images_embeddings = model.image_pair_encoder(image1, image2)

    idx = torch.full((1, 1), model.bos_token_id, dtype=torch.long, device=image1.device)
    step_infos: List[Dict[str, object]] = []
    stop_reason = "max_gen_len"

    for step_index in range(max_gen_len):
        logits, _ = model.transform_decoder(idx=idx, images_embeddings=images_embeddings)
        next_logits = logits[:, -1, :].clone()
        next_logits[:, PAD_TOKEN_ID] = -float("inf")
        next_logits[:, START_TOKEN_ID] = -float("inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token_id = int(torch.argmax(probs, dim=-1).item())
        next_token_prob = float(probs[0, next_token_id].item())

        top_k = min(5, probs.shape[-1])
        top_probs, top_ids = torch.topk(probs[0], k=top_k)
        step_infos.append(
            {
                "step": step_index + 1,
                "token_id": next_token_id,
                "token": tokenizer.decode([next_token_id], skip_special_tokens=False)[0],
                "probability": next_token_prob,
                "top_candidates": [
                    {
                        "token_id": int(candidate_id.item()),
                        "token": tokenizer.decode([int(candidate_id.item())], skip_special_tokens=False)[0],
                        "probability": float(candidate_prob.item()),
                    }
                    for candidate_prob, candidate_id in zip(top_probs, top_ids)
                ],
            }
        )

        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=image1.device)
        idx = torch.cat([idx, next_token_tensor], dim=1)

        if next_token_id == END_TOKEN_ID:
            stop_reason = "eos"
            break

    full_ids = idx[0].tolist()
    full_tokens = tokenizer.decode(full_ids, skip_special_tokens=False)
    decoded_tokens = tokenizer.decode(full_ids, skip_special_tokens=True)
    chosen_probabilities = [step["probability"] for step in step_infos]
    mean_probability = sum(chosen_probabilities) / len(chosen_probabilities) if chosen_probabilities else None

    return {
        "full_ids": full_ids,
        "full_tokens": full_tokens,
        "decoded_tokens": decoded_tokens,
        "step_infos": step_infos,
        "stop_reason": stop_reason,
        "mean_probability": mean_probability,
    }


def print_verbose_trace(step_infos: List[Dict[str, object]]) -> None:
    print("Generation trace:")
    print("  prefix: [START]")
    for step in step_infos:
        top_candidates = ", ".join(
            f"{candidate['token']}({candidate['probability']:.4f})"
            for candidate in step["top_candidates"]
        )
        print(
            f"  step {step['step']}: token_id={step['token_id']} token={step['token']} "
            f"prob={step['probability']:.4f} top=[{top_candidates}]"
        )


def print_probe_result(
    target_name: str,
    target_tokens: List[str],
    result: Dict[str, object],
    interpretation: Dict[str, object],
) -> None:
    print("=" * 80)
    print(f"target rotation: {target_name}")
    print(f"target tokens: {target_tokens}")
    print(f"predicted token ids: {result['full_ids']}")
    print(f"predicted tokens: {result['full_tokens']}")
    print(f"decoded chain: {interpretation['decoded_chain']}")
    print(f"canonical D4 element: {interpretation['canonical_element']}")
    print(f"canonical tokens: {interpretation['canonical_tokens']}")
    print(f"is canonical: {interpretation['is_canonical']}")
    if result["mean_probability"] is not None:
        print(f"mean chosen-token probability: {result['mean_probability']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a D4 checkpoint on synthetic rotations of a single image.")
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
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=None,
        help="Maximum number of autoregressive decoding steps. Defaults to config max_seq_len - 1.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step token generation details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    image1_path = _resolve_project_path(args.image1)

    config = OmegaConf.load(config_path)
    device = choose_device(args.device, config)

    model = ImageTransformPredictor(config.model)
    model.to(device)
    model.eval()

    load_model_checkpoint(model, checkpoint_path, device, config_path)

    tokenizer = TransformTokenizer()
    preprocess = model.image_pair_encoder.preprocess
    base_image = load_pil_image(image1_path)
    image1_tensor = pil_to_tensor(base_image, preprocess, device)

    max_gen_len = args.max_gen_len or (config.model.decoder.max_seq_len - 1)

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"image1: {image1_path}")
    print(f"encoder: {config.model.encoder.type}")
    print("rotation probe: e, r, r2, r3")

    for target_name, target_tokens in ROTATION_SEQUENCES.items():
        transformed_image = apply_rotation_sequence(base_image, target_tokens)
        image2_tensor = pil_to_tensor(transformed_image, preprocess, device)

        result = autoregressive_decode(
            model=model,
            image1=image1_tensor,
            image2=image2_tensor,
            tokenizer=tokenizer,
            max_gen_len=max_gen_len,
        )
        interpretation = canonicalize_prediction(result["decoded_tokens"])

        print_probe_result(
            target_name=target_name,
            target_tokens=target_tokens,
            result=result,
            interpretation=interpretation,
        )
        if args.verbose:
            print_verbose_trace(result["step_infos"])


if __name__ == "__main__":
    main()
