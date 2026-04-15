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
    END_TOKEN_ID,
    PAD_TOKEN_ID,
    START_TOKEN_ID,
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
    if path.exists():
        return str(path.resolve())
    return str(PROJECT_ROOT / path)


def _resolve_output_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    if path.parent != Path(".") and (PROJECT_ROOT / path.parent).exists():
        return str(PROJECT_ROOT / path)
    return str(path.resolve())


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


def _apply_generator_to_matrix(
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
            matrix = _apply_generator_to_matrix(matrix, token)
        lookup[matrix] = (element_name, list(tokens))
    return lookup


CANONICAL_LOOKUP = _build_canonical_lookup()


def canonicalize_d4_tokens(tokens: List[str]) -> Tuple[str, List[str]]:
    matrix = IDENTITY_MATRIX
    for token in tokens:
        if token not in {"e", "r", "s"}:
            raise ValueError(f"Unsupported D4 token: {token}")
        matrix = _apply_generator_to_matrix(matrix, token)
    return CANONICAL_LOOKUP[matrix]


def inverse_d4_tokens(tokens: List[str]) -> Tuple[str, List[str]]:
    inverse_sequence: List[str] = []
    for token in reversed(tokens):
        if token == "e":
            inverse_sequence.append("e")
        elif token == "r":
            inverse_sequence.extend(["r", "r", "r"])
        elif token == "s":
            inverse_sequence.append("s")
        else:
            raise ValueError(f"Unsupported D4 token: {token}")
    return canonicalize_d4_tokens(inverse_sequence)


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


def format_tokens(tokens: Optional[List[str]]) -> str:
    if tokens is None:
        return "-"
    if not tokens:
        return "e"
    return " ".join(tokens)


def format_bool(value: bool) -> str:
    return "да" if value else "нет"


def choose_device(device_arg: Optional[str], config) -> torch.device:
    requested = device_arg or config.training.get("device", "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA недоступна, переключаюсь на CPU.")
        return torch.device("cpu")
    return torch.device(requested)


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
            "Не удалось загрузить checkpoint в модель. "
            "Скорее всего, checkpoint и config не подходят друг другу.\n"
            f"config: {config_path}\n"
            f"checkpoint: {checkpoint_path}\n"
            f"encoder.type в config: {encoder_type}\n"
            "Для d4_efficientnet используй train_config_d4.yaml, "
            "для d4_vit используй train_config_d4_vit.yaml."
        ) from exc


def interpret_completion_tokens(tokens: List[str]) -> Dict[str, object]:
    if tokens == ["[NULL]"]:
        return {
            "status": "unrelated",
            "is_null": True,
            "decoded_chain": "∅",
            "completion_element": "∅",
            "completion_tokens": ["[NULL]"],
            "implied_applied_element": "∅",
            "implied_applied_tokens": ["[NULL]"],
        }

    if "[NULL]" in tokens:
        return {
            "status": "invalid_mixed_null_sequence",
            "is_null": True,
            "decoded_chain": " ".join(tokens),
            "completion_element": "invalid",
            "completion_tokens": None,
            "implied_applied_element": "invalid",
            "implied_applied_tokens": None,
        }

    try:
        completion_element, completion_tokens = canonicalize_d4_tokens(tokens)
        applied_element, applied_tokens = inverse_d4_tokens(completion_tokens)
    except ValueError:
        return {
            "status": "invalid_unknown_token",
            "is_null": False,
            "decoded_chain": " ".join(tokens) if tokens else "(empty)",
            "completion_element": "invalid",
            "completion_tokens": None,
            "implied_applied_element": "invalid",
            "implied_applied_tokens": None,
        }

    return {
        "status": "related",
        "is_null": False,
        "decoded_chain": " ".join(tokens) if tokens else "(empty)",
        "completion_element": completion_element,
        "completion_tokens": completion_tokens,
        "implied_applied_element": applied_element,
        "implied_applied_tokens": applied_tokens,
        "is_canonical": tokens == completion_tokens,
    }


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
        logits, _ = model.transform_decoder(idx=idx, images_embeddings=images_embeddings, targets=None)
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

        idx = torch.cat(
            [idx, torch.tensor([[next_token_id]], dtype=torch.long, device=image1.device)],
            dim=1,
        )
        if next_token_id == END_TOKEN_ID:
            stop_reason = "eos"
            break

    full_ids = idx[0].tolist()
    full_tokens = tokenizer.decode(full_ids, skip_special_tokens=False)
    decoded_tokens = tokenizer.decode(full_ids, skip_special_tokens=True)
    chosen_probs = [step["probability"] for step in step_infos]

    return {
        "full_ids": full_ids,
        "full_tokens": full_tokens,
        "decoded_tokens": decoded_tokens,
        "step_infos": step_infos,
        "stop_reason": stop_reason,
        "mean_probability": sum(chosen_probs) / len(chosen_probs) if chosen_probs else None,
    }


def print_verbose_trace(step_infos: List[Dict[str, object]]) -> None:
    print("Пошаговая генерация:")
    print("  старт: [START]")
    for step in step_infos:
        top_candidates = ", ".join(
            f"{candidate['token']}({candidate['probability']:.4f})"
            for candidate in step["top_candidates"]
        )
        print(
            f"  шаг {step['step']}: id={step['token_id']} токен={step['token']} "
            f"p={step['probability']:.4f} top=[{top_candidates}]"
        )


def print_result(
    result: Dict[str, object],
    interpretation: Dict[str, object],
    image1_path: str,
    image2_label: str,
    expected_completion: Optional[Tuple[str, List[str]]] = None,
    show_paths: bool = True,
    show_details: bool = False,
) -> None:
    if show_paths:
        print(f"I1: {image1_path}")
        print(f"I2: {image2_label}")
    if expected_completion is not None:
        print(f"Ожидалось h: {expected_completion[0]} = {format_tokens(expected_completion[1])}")

    if interpretation["is_null"]:
        print("Ответ: ∅")
        print("Итог: модель считает пару несвязанной.")
    else:
        print(
            f"Ответ h: {interpretation['completion_element']} = "
            f"{format_tokens(interpretation['completion_tokens'])}"
        )
        print(
            f"Значит исходное g: {interpretation['implied_applied_element']} = "
            f"{format_tokens(interpretation['implied_applied_tokens'])}"
        )
        if "is_canonical" in interpretation:
            print(f"Канонично: {format_bool(interpretation['is_canonical'])}")
            if not interpretation["is_canonical"]:
                print(f"Сырая цепочка: {interpretation['decoded_chain']}")

    if result["mean_probability"] is not None:
        print(f"Средняя p: {result['mean_probability']:.4f}")

    if show_details:
        print(f"token ids: {result['full_ids']}")
        print(f"tokens: {result['full_tokens']}")


def build_all_d4_pairs(base_image: Image.Image) -> List[Tuple[str, List[str], Image.Image]]:
    pairs: List[Tuple[str, List[str], Image.Image]] = []
    for element_name, tokens in D4_CANONICAL_SEQUENCES.items():
        pairs.append((element_name, list(tokens), apply_sequence_to_pil(base_image, tokens)))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Инференс D4 completion-модели.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Путь к YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к checkpoint модели.")
    parser.add_argument("--image1", type=str, required=True, help="Исходная картинка I1.")
    parser.add_argument("--image2", type=str, default=None, help="Вторая картинка I2. Если не задана, будет D4-sweep.")
    parser.add_argument("--device", type=str, default=None, help="Например: cuda, cuda:0 или cpu.")
    parser.add_argument("--max_gen_len", type=int, default=None, help="Максимальная длина генерации без BOS.")
    parser.add_argument("--verbose", action="store_true", help="Показать token ids и пошаговую генерацию.")
    parser.add_argument(
        "--restore_output",
        type=str,
        default=None,
        help="Куда сохранить h(I2). Используется только вместе с --image2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    image1_path = _resolve_project_path(args.image1)
    image2_path = _resolve_project_path(args.image2)

    config = OmegaConf.load(config_path)
    device = choose_device(args.device, config)

    model = ImageTransformPredictor(config.model)
    model.to(device)
    model.eval()
    load_model_checkpoint(model, checkpoint_path, device, config_path)

    tokenizer = TransformTokenizer()
    preprocess = model.image_pair_encoder.preprocess
    max_gen_len = args.max_gen_len or (config.model.decoder.max_seq_len - 1)

    base_image = load_pil_image(image1_path)
    image1_tensor = pil_to_tensor(base_image, preprocess, device)

    print(f"Модель: {config.model.encoder.type}")
    print("Ответ модели: h, где h(I2) должно совпасть с I1.")
    if args.verbose:
        print(f"config: {config_path}")
        print(f"checkpoint: {checkpoint_path}")

    if image2_path is not None:
        image2 = load_pil_image(image2_path)
        image2_tensor = pil_to_tensor(image2, preprocess, device)
        result = autoregressive_decode(model, image1_tensor, image2_tensor, tokenizer, max_gen_len)
        interpretation = interpret_completion_tokens(result["decoded_tokens"])

        print("=" * 64)
        print("Режим: пара картинок")
        print_result(result, interpretation, image1_path, image2_path, show_details=args.verbose)
        if args.verbose:
            print_verbose_trace(result["step_infos"])

        if args.restore_output and not interpretation["is_null"] and interpretation["completion_tokens"]:
            restored = apply_sequence_to_pil(image2, interpretation["completion_tokens"])
            restore_output = _resolve_output_path(args.restore_output)
            Path(restore_output).parent.mkdir(parents=True, exist_ok=True)
            restored.save(restore_output)
            print(f"Сохранено h(I2): {restore_output}")
        return

    print("Режим: одна картинка. Создаю 8 D4-вариантов и проверяю модель.")
    for pair_index, (applied_element, applied_tokens, image2) in enumerate(build_all_d4_pairs(base_image), start=1):
        image2_tensor = pil_to_tensor(image2, preprocess, device)
        expected_completion = inverse_d4_tokens(applied_tokens)
        result = autoregressive_decode(model, image1_tensor, image2_tensor, tokenizer, max_gen_len)
        interpretation = interpret_completion_tokens(result["decoded_tokens"])

        print("=" * 64)
        print(f"{pair_index}/8. К I1 применили g: {applied_element} = {format_tokens(applied_tokens)}")
        print_result(
            result=result,
            interpretation=interpretation,
            image1_path=image1_path,
            image2_label=f"<generated:{applied_element}>",
            expected_completion=expected_completion,
            show_paths=False,
            show_details=args.verbose,
        )
        if args.verbose:
            print_verbose_trace(result["step_infos"])


if __name__ == "__main__":
    main()
