#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm import tqdm

from src.dataset.augmentation import canonicalize_d4_sequence, inverse_d4_sequence
from src.dataset.tokenizer import (
    D4_CANONICAL_SEQUENCES,
    END_TOKEN_ID,
    NULL_SEQUENCE,
    PAD_TOKEN_ID,
    START_TOKEN_ID,
    TransformTokenizer,
)
from src.model import ImageTransformPredictor


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NULL_LABEL = "∅"
INVALID_LABEL = "invalid"
CLASS_LABELS = list(D4_CANONICAL_SEQUENCES.keys()) + [NULL_LABEL, INVALID_LABEL]
SEQUENCE_TO_ELEMENT = {tuple(seq): name for name, seq in D4_CANONICAL_SEQUENCES.items()}


@dataclass
class ModelSpec:
    name: str
    config_path: str
    checkpoint_path: str


@dataclass
class ImageItem:
    path: str
    domain: str


@dataclass
class EvalRecord:
    image1_path: str
    domain: str
    applied_element: str
    applied_tokens: List[str]
    is_negative: bool = False
    image2_path: Optional[str] = None


@dataclass
class BatchData:
    image1: torch.Tensor
    image2: torch.Tensor
    target_ids: torch.Tensor
    target_tokens: List[List[str]]
    target_labels: List[str]
    applied_labels: List[str]
    domains: List[str]
    records: List[EvalRecord]


@dataclass
class EvalStats:
    count: int = 0
    loss_sum: float = 0.0
    teacher_seq_correct: int = 0
    teacher_token_correct: int = 0
    teacher_token_total: int = 0
    generated_seq_correct: int = 0
    generated_token_correct: int = 0
    generated_token_total: int = 0
    element_correct: int = 0
    applied_correct: int = 0
    canonical_correct: int = 0
    generated_confidence_sum: float = 0.0
    generated_confidence_count: int = 0
    target_confidence_sum: float = 0.0
    target_confidence_count: int = 0
    confusion: Counter = field(default_factory=Counter)
    applied_confusion: Counter = field(default_factory=Counter)

    def update(
        self,
        loss: float,
        teacher_seq_correct: bool,
        teacher_token_correct: int,
        teacher_token_total: int,
        generated_seq_correct: bool,
        generated_token_correct: int,
        generated_token_total: int,
        element_correct: bool,
        applied_correct: bool,
        canonical_correct: bool,
        generated_confidence: Optional[float],
        target_confidence: Optional[float],
        target_label: str,
        predicted_label: str,
        applied_label: str,
        predicted_applied_label: str,
    ) -> None:
        self.count += 1
        self.loss_sum += loss
        self.teacher_seq_correct += int(teacher_seq_correct)
        self.teacher_token_correct += teacher_token_correct
        self.teacher_token_total += teacher_token_total
        self.generated_seq_correct += int(generated_seq_correct)
        self.generated_token_correct += generated_token_correct
        self.generated_token_total += generated_token_total
        self.element_correct += int(element_correct)
        self.applied_correct += int(applied_correct)
        self.canonical_correct += int(canonical_correct)
        if generated_confidence is not None and not math.isnan(generated_confidence):
            self.generated_confidence_sum += generated_confidence
            self.generated_confidence_count += 1
        if target_confidence is not None and not math.isnan(target_confidence):
            self.target_confidence_sum += target_confidence
            self.target_confidence_count += 1
        self.confusion[(target_label, predicted_label)] += 1
        self.applied_confusion[(applied_label, predicted_applied_label)] += 1


@dataclass
class ContinuationStats:
    count: int = 0
    next_token_correct: int = 0
    continuation_correct: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0

    def update(self, next_correct: bool, exact_correct: bool, confidence: Optional[float]) -> None:
        self.count += 1
        self.next_token_correct += int(next_correct)
        self.continuation_correct += int(exact_correct)
        if confidence is not None and not math.isnan(confidence):
            self.confidence_sum += confidence
            self.confidence_count += 1


def resolve_input_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return str(PROJECT_ROOT / path)


def resolve_output_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    if path.parent != Path(".") and (PROJECT_ROOT / path.parent).exists():
        return str(PROJECT_ROOT / path)
    return str(path.resolve())


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def format_tokens(tokens: Optional[Sequence[str]]) -> str:
    if tokens is None:
        return "-"
    if not tokens:
        return "e"
    return " ".join(tokens)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def apply_token_to_image(image: Image.Image, token: str) -> Image.Image:
    if token == "e":
        return image
    if token == "r":
        return image.rotate(90, expand=True)
    if token == "s":
        return ImageOps.mirror(image)
    raise ValueError(f"Unknown D4 token: {token}")


def apply_sequence_to_image(image: Image.Image, tokens: Sequence[str]) -> Image.Image:
    result = image.copy()
    for token in tokens:
        result = apply_token_to_image(result, token)
    return result.convert("RGB")


def canonical_element_from_tokens(tokens: Sequence[str]) -> Tuple[str, List[str]]:
    canonical_tokens = canonicalize_d4_sequence(list(tokens))
    return SEQUENCE_TO_ELEMENT[tuple(canonical_tokens)], canonical_tokens


def target_tokens_from_applied(applied_tokens: Sequence[str], target_mode: str) -> List[str]:
    if target_mode == "applied":
        return list(applied_tokens)
    if target_mode in ("completion", "inverse"):
        return inverse_d4_sequence(list(applied_tokens))
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def decode_label(tokens: Sequence[str]) -> Dict[str, object]:
    tokens = list(tokens)
    if tokens == list(NULL_SEQUENCE):
        return {
            "label": NULL_LABEL,
            "canonical_tokens": list(NULL_SEQUENCE),
            "is_null": True,
            "is_valid": True,
            "is_canonical": True,
        }
    if "[NULL]" in tokens:
        return {
            "label": INVALID_LABEL,
            "canonical_tokens": None,
            "is_null": False,
            "is_valid": False,
            "is_canonical": False,
        }
    if any(token not in {"e", "r", "s"} for token in tokens):
        return {
            "label": INVALID_LABEL,
            "canonical_tokens": None,
            "is_null": False,
            "is_valid": False,
            "is_canonical": False,
        }

    label, canonical_tokens = canonical_element_from_tokens(tokens)
    return {
        "label": label,
        "canonical_tokens": canonical_tokens,
        "is_null": False,
        "is_valid": True,
        "is_canonical": tokens == canonical_tokens,
    }


def implied_applied_label(decoded: Dict[str, object], target_mode: str) -> str:
    if decoded["label"] in (NULL_LABEL, INVALID_LABEL):
        return decoded["label"]
    canonical_tokens = decoded["canonical_tokens"]
    if not isinstance(canonical_tokens, list):
        return INVALID_LABEL
    if target_mode == "applied":
        return str(decoded["label"])
    applied_tokens = inverse_d4_sequence(canonical_tokens)
    applied_label, _ = canonical_element_from_tokens(applied_tokens)
    return applied_label


def trim_at_pad(ids: Sequence[int]) -> List[int]:
    trimmed = []
    for token_id in ids:
        if token_id == PAD_TOKEN_ID:
            break
        trimmed.append(int(token_id))
    return trimmed


def trim_generated_ids(ids: Sequence[int]) -> List[int]:
    trimmed = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == PAD_TOKEN_ID:
            break
        trimmed.append(token_id)
        if token_id == END_TOKEN_ID:
            break
    return trimmed


def compare_generated_to_expected(generated_ids: Sequence[int], expected_ids: Sequence[int]) -> Tuple[bool, int, int]:
    generated = trim_generated_ids(generated_ids)
    expected = trim_at_pad(expected_ids)
    seq_correct = generated == expected

    generated_y = generated[1:]
    expected_y = expected[1:]
    total = max(len(generated_y), len(expected_y))
    correct = 0
    for position in range(total):
        pred_id = generated_y[position] if position < len(generated_y) else None
        target_id = expected_y[position] if position < len(expected_y) else None
        correct += int(pred_id == target_id)
    return seq_correct, correct, total


def collect_val_items(
    data_dir: str,
    val_size: float,
    random_seed: int,
    max_images_per_domain: Optional[int],
) -> List[ImageItem]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_dir}")

    domain_to_paths: Dict[str, List[str]] = {}
    direct_images = [str(path) for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if direct_images:
        domain_to_paths["root"] = direct_images

    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        paths = [
            str(path)
            for path in sorted(domain_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if paths:
            domain_to_paths[domain_dir.name] = paths

    rng = random.Random(random_seed)
    val_items: List[ImageItem] = []
    for domain, paths in domain_to_paths.items():
        shuffled = rng.sample(paths, len(paths))
        n_val = int(len(shuffled) * val_size)
        selected = shuffled[:n_val]
        if max_images_per_domain is not None:
            selected = selected[:max_images_per_domain]
        val_items.extend(ImageItem(path=path, domain=domain) for path in selected)

    if not val_items:
        raise ValueError(
            "Validation split is empty. Increase --val_size, remove --max_images_per_domain, "
            "or check the data directory."
        )
    return val_items


def build_eval_records(
    val_items: Sequence[ImageItem],
    allowed_elements: Sequence[str],
    negative_pairs_per_image: int,
) -> List[EvalRecord]:
    invalid = [element for element in allowed_elements if element not in D4_CANONICAL_SEQUENCES]
    if invalid:
        raise ValueError(f"Unknown D4 elements in --allowed_elements: {invalid}")

    records: List[EvalRecord] = []
    for item in val_items:
        for element in allowed_elements:
            records.append(
                EvalRecord(
                    image1_path=item.path,
                    domain=item.domain,
                    applied_element=element,
                    applied_tokens=list(D4_CANONICAL_SEQUENCES[element]),
                )
            )

    if negative_pairs_per_image > 0 and len(val_items) > 1:
        total_items = len(val_items)
        for item_index, item in enumerate(val_items):
            for offset in range(1, negative_pairs_per_image + 1):
                negative_item = val_items[(item_index + offset) % total_items]
                if negative_item.path == item.path:
                    negative_item = val_items[(item_index + offset + 1) % total_items]
                records.append(
                    EvalRecord(
                        image1_path=item.path,
                        image2_path=negative_item.path,
                        domain=item.domain,
                        applied_element=NULL_LABEL,
                        applied_tokens=list(NULL_SEQUENCE),
                        is_negative=True,
                    )
                )
    return records


def chunks(items: Sequence[EvalRecord], batch_size: int) -> Iterable[List[EvalRecord]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def make_batch(
    records: Sequence[EvalRecord],
    preprocess,
    tokenizer: TransformTokenizer,
    target_mode: str,
    max_seq_len: int,
    device: torch.device,
) -> BatchData:
    image1_tensors = []
    image2_tensors = []
    target_ids = []
    target_tokens_list: List[List[str]] = []
    target_labels: List[str] = []
    applied_labels: List[str] = []
    domains: List[str] = []

    for record in records:
        image1 = load_image(record.image1_path)
        if record.is_negative:
            if record.image2_path is None:
                raise ValueError("Negative record must contain image2_path.")
            image2 = load_image(record.image2_path)
            target_tokens = list(NULL_SEQUENCE)
            target_label = NULL_LABEL
            applied_label = NULL_LABEL
        else:
            image2 = apply_sequence_to_image(image1, record.applied_tokens)
            target_tokens = target_tokens_from_applied(record.applied_tokens, target_mode)
            target_label, _ = canonical_element_from_tokens(target_tokens)
            applied_label = record.applied_element

        image1_tensors.append(preprocess(image1))
        image2_tensors.append(preprocess(image2))
        target_ids.append(
            tokenizer.encode(
                transforms=target_tokens,
                add_special_tokens=True,
                max_seq_len=max_seq_len,
                return_targets=False,
            )
        )
        target_tokens_list.append(target_tokens)
        target_labels.append(target_label)
        applied_labels.append(applied_label)
        domains.append(record.domain)

    return BatchData(
        image1=torch.stack(image1_tensors).to(device),
        image2=torch.stack(image2_tensors).to(device),
        target_ids=torch.stack(target_ids).to(device),
        target_tokens=target_tokens_list,
        target_labels=target_labels,
        applied_labels=applied_labels,
        domains=domains,
        records=list(records),
    )


def choose_device(device_arg: Optional[str], config) -> torch.device:
    requested = device_arg or config.training.get("device", "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA недоступна, переключаюсь на CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_checkpoint(model: ImageTransformPredictor, checkpoint_path: str, device: torch.device, config_path: str) -> None:
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


@torch.no_grad()
def greedy_decode_from_embeddings(
    model: ImageTransformPredictor,
    images_embeddings: torch.Tensor,
    max_new_tokens: int,
    prefix_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[Optional[float]]]:
    batch_size = images_embeddings.shape[0]
    device = images_embeddings.device
    if prefix_ids is None:
        idx = torch.full((batch_size, 1), model.bos_token_id, dtype=torch.long, device=device)
    else:
        idx = prefix_ids.to(device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    confidence_values: List[List[float]] = [[] for _ in range(batch_size)]

    for _ in range(max_new_tokens):
        logits, _ = model.transform_decoder(idx=idx, images_embeddings=images_embeddings, targets=None)
        next_logits = logits[:, -1, :].clone()
        next_logits[:, PAD_TOKEN_ID] = -float("inf")
        next_logits[:, START_TOKEN_ID] = -float("inf")

        probs = F.softmax(next_logits, dim=-1)
        next_ids = probs.argmax(dim=-1)
        next_probs = probs.gather(1, next_ids.unsqueeze(1)).squeeze(1)

        active = ~finished
        for sample_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            confidence_values[sample_index].append(float(next_probs[sample_index].item()))

        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_TOKEN_ID), next_ids)
        finished = finished | (next_ids == END_TOKEN_ID)
        idx = torch.cat([idx, next_ids.unsqueeze(1)], dim=1)
        if finished.all():
            break

    target_len = (prefix_ids.shape[1] if prefix_ids is not None else 1) + max_new_tokens
    if idx.shape[1] < target_len:
        pad = torch.full((batch_size, target_len - idx.shape[1]), PAD_TOKEN_ID, dtype=torch.long, device=device)
        idx = torch.cat([idx, pad], dim=1)

    mean_confidences: List[Optional[float]] = []
    for values in confidence_values:
        mean_confidences.append(sum(values) / len(values) if values else None)
    return idx[:, :target_len], mean_confidences


def compute_teacher_metrics(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_token_id: int,
) -> Dict[str, object]:
    targets = torch.full_like(target_ids, pad_token_id)
    targets[:, :-1] = target_ids[:, 1:]

    pred_ids = logits.argmax(dim=-1)
    mask = targets != pad_token_id
    correct = (pred_ids == targets) & mask
    seq_correct = ((pred_ids == targets) | (~mask)).all(dim=1)

    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).view_as(targets)

    probs = F.softmax(logits, dim=-1)
    safe_targets = targets.clamp(min=0)
    target_probs = probs.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)

    per_sample_losses = []
    per_sample_confidences = []
    for sample_index in range(target_ids.shape[0]):
        sample_mask = mask[sample_index]
        token_count = int(sample_mask.sum().item())
        if token_count == 0:
            per_sample_losses.append(0.0)
            per_sample_confidences.append(None)
            continue
        per_sample_losses.append(float(token_losses[sample_index][sample_mask].mean().item()))
        per_sample_confidences.append(float(target_probs[sample_index][sample_mask].mean().item()))

    return {
        "targets": targets,
        "seq_correct": seq_correct.cpu().tolist(),
        "token_correct": correct.sum(dim=1).cpu().tolist(),
        "token_total": mask.sum(dim=1).cpu().tolist(),
        "loss": per_sample_losses,
        "target_confidence": per_sample_confidences,
    }


def decoded_tokens_from_generated(tokenizer: TransformTokenizer, generated_ids: Sequence[int]) -> List[str]:
    trimmed = trim_generated_ids(generated_ids)
    return tokenizer.decode(trimmed, skip_special_tokens=True)


def evaluate_continuations(
    model: ImageTransformPredictor,
    images_embeddings: torch.Tensor,
    target_ids: torch.Tensor,
    target_tokens: Sequence[Sequence[str]],
    domains: Sequence[str],
    tokenizer: TransformTokenizer,
    continuation_by_prefix: Dict[int, ContinuationStats],
    continuation_by_domain_prefix: Dict[Tuple[str, int], ContinuationStats],
    max_prefix_len: int,
) -> None:
    del tokenizer
    target_core_ids: List[List[int]] = []
    for ids in target_ids.detach().cpu().tolist():
        trimmed = trim_at_pad(ids)
        core = []
        for token_id in trimmed[1:]:
            if token_id == END_TOKEN_ID:
                break
            core.append(token_id)
        target_core_ids.append(core)

    max_observed_prefix = min(max_prefix_len, max(len(tokens) for tokens in target_core_ids))
    for prefix_len in range(max_observed_prefix + 1):
        sample_indices = [idx for idx, tokens in enumerate(target_core_ids) if prefix_len <= len(tokens)]
        if not sample_indices:
            continue

        prefix_rows = []
        expected_remaining_rows = []
        for sample_index in sample_indices:
            core = target_core_ids[sample_index]
            prefix_rows.append([START_TOKEN_ID] + core[:prefix_len])
            expected_remaining_rows.append(core[prefix_len:] + [END_TOKEN_ID])

        prefix_tensor = torch.tensor(prefix_rows, dtype=torch.long, device=images_embeddings.device)
        subset_embeddings = images_embeddings[sample_indices]
        max_new_tokens = model.config.decoder.max_seq_len - prefix_tensor.shape[1]
        generated_ids, confidences = greedy_decode_from_embeddings(
            model=model,
            images_embeddings=subset_embeddings,
            max_new_tokens=max_new_tokens,
            prefix_ids=prefix_tensor,
        )

        generated_cpu = generated_ids.detach().cpu().tolist()
        for local_index, sample_index in enumerate(sample_indices):
            generated_remaining = trim_generated_ids(generated_cpu[local_index][prefix_tensor.shape[1]:])
            expected_remaining = expected_remaining_rows[local_index]
            next_correct = bool(generated_remaining) and generated_remaining[0] == expected_remaining[0]
            exact_correct = generated_remaining == expected_remaining
            confidence = confidences[local_index]

            continuation_by_prefix[prefix_len].update(next_correct, exact_correct, confidence)
            continuation_by_domain_prefix[(domains[sample_index], prefix_len)].update(next_correct, exact_correct, confidence)


def update_stats_from_batch(
    model_name: str,
    target_mode: str,
    tokenizer: TransformTokenizer,
    batch: BatchData,
    teacher_metrics: Dict[str, object],
    generated_ids: torch.Tensor,
    generated_confidences: List[Optional[float]],
    stats_overall: EvalStats,
    stats_by_domain: Dict[str, EvalStats],
    prediction_rows: Optional[List[Dict[str, object]]],
) -> None:
    generated_cpu = generated_ids.detach().cpu().tolist()
    target_cpu = batch.target_ids.detach().cpu().tolist()

    for sample_index, generated_sample_ids in enumerate(generated_cpu):
        predicted_tokens = decoded_tokens_from_generated(tokenizer, generated_sample_ids)
        decoded = decode_label(predicted_tokens)
        predicted_label = str(decoded["label"])
        predicted_applied_label = implied_applied_label(decoded, target_mode)

        expected_seq_correct, generated_token_correct, generated_token_total = compare_generated_to_expected(
            generated_sample_ids,
            target_cpu[sample_index],
        )

        target_label = batch.target_labels[sample_index]
        applied_label = batch.applied_labels[sample_index]
        teacher_seq_correct = bool(teacher_metrics["seq_correct"][sample_index])
        teacher_token_correct = int(teacher_metrics["token_correct"][sample_index])
        teacher_token_total = int(teacher_metrics["token_total"][sample_index])
        loss = float(teacher_metrics["loss"][sample_index])
        target_confidence = teacher_metrics["target_confidence"][sample_index]
        generated_confidence = generated_confidences[sample_index]

        kwargs = dict(
            loss=loss,
            teacher_seq_correct=teacher_seq_correct,
            teacher_token_correct=teacher_token_correct,
            teacher_token_total=teacher_token_total,
            generated_seq_correct=expected_seq_correct,
            generated_token_correct=generated_token_correct,
            generated_token_total=generated_token_total,
            element_correct=predicted_label == target_label,
            applied_correct=predicted_applied_label == applied_label,
            canonical_correct=bool(decoded["is_canonical"]),
            generated_confidence=generated_confidence,
            target_confidence=target_confidence,
            target_label=target_label,
            predicted_label=predicted_label,
            applied_label=applied_label,
            predicted_applied_label=predicted_applied_label,
        )
        stats_overall.update(**kwargs)
        stats_by_domain[batch.domains[sample_index]].update(**kwargs)

        if prediction_rows is not None:
            record = batch.records[sample_index]
            prediction_rows.append(
                {
                    "model": model_name,
                    "domain": batch.domains[sample_index],
                    "image1": record.image1_path,
                    "image2": record.image2_path or f"<generated:{record.applied_element}>",
                    "is_negative": record.is_negative,
                    "applied_element": applied_label,
                    "target_label": target_label,
                    "target_tokens": format_tokens(batch.target_tokens[sample_index]),
                    "predicted_label": predicted_label,
                    "predicted_tokens_raw": format_tokens(predicted_tokens),
                    "predicted_tokens_canonical": format_tokens(decoded["canonical_tokens"]),
                    "predicted_applied_label": predicted_applied_label,
                    "teacher_seq_correct": teacher_seq_correct,
                    "generated_seq_correct": expected_seq_correct,
                    "element_correct": predicted_label == target_label,
                    "applied_correct": predicted_applied_label == applied_label,
                    "generated_confidence": generated_confidence,
                    "target_confidence": target_confidence,
                }
            )


def classification_metrics(confusion: Counter, labels: Sequence[str]) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    true_counts = Counter()
    pred_counts = Counter()
    for (true_label, pred_label), value in confusion.items():
        true_counts[true_label] += value
        pred_counts[pred_label] += value

    active_labels = [
        label
        for label in labels
        if true_counts[label] > 0 or pred_counts[label] > 0
    ]
    if not active_labels:
        active_labels = list(labels)

    per_class_rows = []
    macro_precision_values = []
    macro_recall_values = []
    macro_f1_values = []
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0
    total_support = sum(true_counts.values())

    for label in active_labels:
        tp = confusion[(label, label)]
        fp = pred_counts[label] - tp
        fn = true_counts[label] - tp
        support = true_counts[label]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        macro_precision_values.append(precision)
        macro_recall_values.append(recall)
        macro_f1_values.append(f1)
        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support
        per_class_rows.append(
            {
                "class": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted": pred_counts[label],
            }
        )

    summary = {
        "precision_macro": safe_div(sum(macro_precision_values), len(macro_precision_values)),
        "recall_macro": safe_div(sum(macro_recall_values), len(macro_recall_values)),
        "f1_macro": safe_div(sum(macro_f1_values), len(macro_f1_values)),
        "precision_weighted": safe_div(weighted_precision_sum, total_support),
        "recall_weighted": safe_div(weighted_recall_sum, total_support),
        "f1_weighted": safe_div(weighted_f1_sum, total_support),
    }
    return summary, per_class_rows


def stats_to_row(model_name: str, domain: str, stats: EvalStats) -> Dict[str, object]:
    class_summary, _ = classification_metrics(stats.confusion, CLASS_LABELS)
    row = {
        "model": model_name,
        "domain": domain,
        "samples": stats.count,
        "loss": safe_div(stats.loss_sum, stats.count),
        "teacher_seq_acc": safe_div(stats.teacher_seq_correct, stats.count),
        "teacher_token_acc": safe_div(stats.teacher_token_correct, stats.teacher_token_total),
        "generated_seq_acc": safe_div(stats.generated_seq_correct, stats.count),
        "generated_token_acc": safe_div(stats.generated_token_correct, stats.generated_token_total),
        "target_element_acc": safe_div(stats.element_correct, stats.count),
        "applied_transform_acc": safe_div(stats.applied_correct, stats.count),
        "canonical_rate": safe_div(stats.canonical_correct, stats.count),
        "avg_generated_confidence": safe_div(stats.generated_confidence_sum, stats.generated_confidence_count),
        "avg_teacher_target_confidence": safe_div(stats.target_confidence_sum, stats.target_confidence_count),
    }
    row.update(class_summary)
    return row


def per_class_rows(model_name: str, domain: str, stats: EvalStats) -> List[Dict[str, object]]:
    _, rows = classification_metrics(stats.confusion, CLASS_LABELS)
    for row in rows:
        row["model"] = model_name
        row["domain"] = domain
    return rows


def continuation_rows(
    model_name: str,
    continuation_by_prefix: Dict[int, ContinuationStats],
    continuation_by_domain_prefix: Dict[Tuple[str, int], ContinuationStats],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    overall_rows = []
    for prefix_len, stats in sorted(continuation_by_prefix.items()):
        overall_rows.append(
            {
                "model": model_name,
                "prefix_len": prefix_len,
                "samples": stats.count,
                "next_token_acc": safe_div(stats.next_token_correct, stats.count),
                "continuation_seq_acc": safe_div(stats.continuation_correct, stats.count),
                "avg_confidence": safe_div(stats.confidence_sum, stats.confidence_count),
            }
        )

    domain_rows = []
    for (domain, prefix_len), stats in sorted(continuation_by_domain_prefix.items()):
        domain_rows.append(
            {
                "model": model_name,
                "domain": domain,
                "prefix_len": prefix_len,
                "samples": stats.count,
                "next_token_acc": safe_div(stats.next_token_correct, stats.count),
                "continuation_seq_acc": safe_div(stats.continuation_correct, stats.count),
                "avg_confidence": safe_div(stats.confidence_sum, stats.confidence_count),
            }
        )
    return overall_rows, domain_rows


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def print_short_summary(model_name: str, row: Dict[str, object]) -> None:
    print(f"\n{model_name}")
    print(f"  samples: {row['samples']}")
    print(f"  generated seq acc: {row['generated_seq_acc']:.4f}")
    print(f"  generated token acc: {row['generated_token_acc']:.4f}")
    print(f"  target element acc: {row['target_element_acc']:.4f}")
    print(f"  applied transform acc: {row['applied_transform_acc']:.4f}")
    print(f"  precision macro: {row['precision_macro']:.4f}")
    print(f"  recall macro: {row['recall_macro']:.4f}")
    print(f"  avg confidence: {row['avg_generated_confidence']:.4f}")


def parse_model_specs(raw_specs: Optional[List[List[str]]]) -> List[ModelSpec]:
    if not raw_specs:
        raise ValueError("Pass at least one --model NAME CONFIG CHECKPOINT.")
    specs = []
    for raw_name, raw_config, raw_checkpoint in raw_specs:
        specs.append(
            ModelSpec(
                name=raw_name,
                config_path=resolve_input_path(raw_config),
                checkpoint_path=resolve_input_path(raw_checkpoint),
            )
        )
    return specs


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"matplotlib недоступен, графики не будут построены: {exc}")
        return None


def get_row_value(rows: Sequence[Dict[str, object]], model: str, domain: str, metric: str) -> float:
    for row in rows:
        if row["model"] == model and row["domain"] == domain:
            return float(row.get(metric, 0.0))
    return 0.0


def grouped_bar_plot(
    plt,
    rows: Sequence[Dict[str, object]],
    models: Sequence[str],
    domains: Sequence[str],
    metric: str,
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    if not domains:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 1.2), 5.5))
    palette = ["#3366CC", "#DC3912", "#109618", "#FF9900"]
    x_positions = list(range(len(domains)))
    width = 0.8 / max(1, len(models))

    for model_index, model_name in enumerate(models):
        values = [get_row_value(rows, model_name, domain, metric) for domain in domains]
        offsets = [x + (model_index - (len(models) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=model_name, color=palette[model_index % len(palette)], alpha=0.9)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(domains, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_continuation(plt, rows: Sequence[Dict[str, object]], models: Sequence[str], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    palette = ["#3366CC", "#DC3912", "#109618", "#FF9900"]
    prefix_lengths = sorted({int(row["prefix_len"]) for row in rows})
    for model_index, model_name in enumerate(models):
        model_rows = {int(row["prefix_len"]): row for row in rows if row["model"] == model_name}
        continuation_values = [float(model_rows[prefix]["continuation_seq_acc"]) for prefix in prefix_lengths]
        next_values = [float(model_rows[prefix]["next_token_acc"]) for prefix in prefix_lengths]
        color = palette[model_index % len(palette)]
        ax.plot(prefix_lengths, continuation_values, marker="o", linewidth=2.4, color=color, label=f"{model_name}: continuation")
        ax.plot(prefix_lengths, next_values, marker="s", linewidth=1.8, linestyle="--", color=color, alpha=0.65, label=f"{model_name}: next token")

    ax.set_title("Продолжение последовательности по длине prefix", fontsize=15, fontweight="bold")
    ax.set_xlabel("Сколько target-токенов уже дано модели")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_class_f1_heatmap(plt, rows: Sequence[Dict[str, object]], models: Sequence[str], output_path: str) -> None:
    labels = [label for label in CLASS_LABELS if label != INVALID_LABEL]
    matrix = []
    for label in labels:
        row_values = []
        for model_name in models:
            match = next(
                (
                    row for row in rows
                    if row["model"] == model_name and row["domain"] == "overall" and row["class"] == label
                ),
                None,
            )
            row_values.append(float(match["f1"]) if match is not None else 0.0)
        matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(max(5.5, len(models) * 1.8), 6.5))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("F1 по D4-классам", fontsize=15, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", color="#111111", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(
    plt,
    confusion: Counter,
    model_name: str,
    output_path: str,
    labels: Sequence[str],
) -> None:
    active_labels = [
        label
        for label in labels
        if any(confusion[(label, pred)] > 0 or confusion[(true, label)] > 0 for pred in labels for true in labels)
    ]
    if not active_labels:
        active_labels = list(labels)

    matrix = []
    for true_label in active_labels:
        matrix.append([confusion[(true_label, pred_label)] for pred_label in active_labels])

    fig, ax = plt.subplots(figsize=(max(7, len(active_labels) * 0.8), max(6, len(active_labels) * 0.65)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion matrix: {model_name}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_xticks(range(len(active_labels)))
    ax.set_xticklabels(active_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(active_labels)))
    ax.set_yticklabels(active_labels)
    max_value = max([max(row) for row in matrix] or [0])
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            text_color = "white" if max_value and value > max_value * 0.55 else "#111111"
            ax.text(x_index, y_index, str(value), ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_plots(
    output_dir: str,
    model_names: Sequence[str],
    domain_rows: Sequence[Dict[str, object]],
    class_rows: Sequence[Dict[str, object]],
    continuation_overall_rows: Sequence[Dict[str, object]],
    model_confusions: Dict[str, Counter],
) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    domains = sorted({row["domain"] for row in domain_rows if row["domain"] != "overall"})

    grouped_bar_plot(
        plt,
        domain_rows,
        model_names,
        domains,
        metric="generated_seq_acc",
        title="SeqAcc генерации по доменам отложенной выборки",
        ylabel="Generated sequence accuracy",
        output_path=str(plots_dir / "domain_generated_seq_acc.png"),
    )
    grouped_bar_plot(
        plt,
        domain_rows,
        model_names,
        domains,
        metric="target_element_acc",
        title="Точность D4-элемента по доменам",
        ylabel="Target element accuracy",
        output_path=str(plots_dir / "domain_target_element_acc.png"),
    )
    grouped_bar_plot(
        plt,
        domain_rows,
        model_names,
        domains,
        metric="avg_generated_confidence",
        title="Средняя уверенность модели по доменам",
        ylabel="Average generated-token confidence",
        output_path=str(plots_dir / "domain_confidence.png"),
    )
    plot_continuation(
        plt,
        continuation_overall_rows,
        model_names,
        output_path=str(plots_dir / "continuation_accuracy.png"),
    )
    plot_class_f1_heatmap(
        plt,
        class_rows,
        model_names,
        output_path=str(plots_dir / "class_f1_heatmap.png"),
    )
    for model_name, confusion in model_confusions.items():
        plot_confusion_matrix(
            plt,
            confusion,
            model_name=model_name,
            output_path=str(plots_dir / f"confusion_{model_name}.png"),
            labels=CLASS_LABELS,
        )


def evaluate_model(
    spec: ModelSpec,
    records: Sequence[EvalRecord],
    output_dir: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    config = OmegaConf.load(spec.config_path)
    device = choose_device(args.device, config)
    batch_size = args.batch_size or int(config.training.get("batch_size", 8))
    max_seq_len = int(config.model.decoder.max_seq_len)
    max_gen_len = args.max_gen_len or (max_seq_len - 1)
    target_mode = config.data.get("target_mode", "completion")

    print("=" * 72)
    print(f"Модель: {spec.name}")
    print(f"  encoder: {config.model.encoder.type}")
    print(f"  target_mode: {target_mode}")
    print(f"  checkpoint: {spec.checkpoint_path}")

    model = ImageTransformPredictor(config.model)
    model.to(device)
    model.eval()
    load_checkpoint(model, spec.checkpoint_path, device, spec.config_path)

    tokenizer = TransformTokenizer()
    preprocess = model.image_pair_encoder.preprocess

    stats_overall = EvalStats()
    stats_by_domain: Dict[str, EvalStats] = defaultdict(EvalStats)
    continuation_by_prefix: Dict[int, ContinuationStats] = defaultdict(ContinuationStats)
    continuation_by_domain_prefix: Dict[Tuple[str, int], ContinuationStats] = defaultdict(ContinuationStats)
    prediction_rows: Optional[List[Dict[str, object]]] = [] if args.save_predictions else None

    progress = tqdm(list(chunks(records, batch_size)), desc=f"Eval {spec.name}")
    with torch.no_grad():
        for record_batch in progress:
            batch = make_batch(
                records=record_batch,
                preprocess=preprocess,
                tokenizer=tokenizer,
                target_mode=target_mode,
                max_seq_len=max_seq_len,
                device=device,
            )

            targets = torch.full_like(batch.target_ids, PAD_TOKEN_ID)
            targets[:, :-1] = batch.target_ids[:, 1:]

            images_embeddings = model.image_pair_encoder(batch.image1, batch.image2)
            logits, _ = model.transform_decoder(idx=batch.target_ids, images_embeddings=images_embeddings, targets=targets)
            teacher_metrics = compute_teacher_metrics(logits, batch.target_ids, PAD_TOKEN_ID)

            generated_ids, generated_confidences = greedy_decode_from_embeddings(
                model=model,
                images_embeddings=images_embeddings,
                max_new_tokens=max_gen_len,
            )

            update_stats_from_batch(
                model_name=spec.name,
                target_mode=target_mode,
                tokenizer=tokenizer,
                batch=batch,
                teacher_metrics=teacher_metrics,
                generated_ids=generated_ids,
                generated_confidences=generated_confidences,
                stats_overall=stats_overall,
                stats_by_domain=stats_by_domain,
                prediction_rows=prediction_rows,
            )

            evaluate_continuations(
                model=model,
                images_embeddings=images_embeddings,
                target_ids=batch.target_ids,
                target_tokens=batch.target_tokens,
                domains=batch.domains,
                tokenizer=tokenizer,
                continuation_by_prefix=continuation_by_prefix,
                continuation_by_domain_prefix=continuation_by_domain_prefix,
                max_prefix_len=args.continuation_max_prefix,
            )

    overall_row = stats_to_row(spec.name, "overall", stats_overall)
    domain_rows = [stats_to_row(spec.name, domain, stats) for domain, stats in sorted(stats_by_domain.items())]
    class_rows = per_class_rows(spec.name, "overall", stats_overall)
    for domain, stats in sorted(stats_by_domain.items()):
        class_rows.extend(per_class_rows(spec.name, domain, stats))
    continuation_overall_rows, continuation_domain_rows = continuation_rows(
        spec.name,
        continuation_by_prefix,
        continuation_by_domain_prefix,
    )

    if prediction_rows is not None:
        write_csv(str(Path(output_dir) / f"predictions_{spec.name}.csv"), prediction_rows)

    print_short_summary(spec.name, overall_row)

    if device.type == "cuda":
        del model
        torch.cuda.empty_cache()

    return {
        "overall_row": overall_row,
        "domain_rows": domain_rows,
        "class_rows": class_rows,
        "continuation_overall_rows": continuation_overall_rows,
        "continuation_domain_rows": continuation_domain_rows,
        "confusion": stats_overall.confusion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка D4-моделей на отложенной выборке.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, required=True, help="Корень DomainNet-like данных.")
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "CONFIG", "CHECKPOINT"),
        required=True,
        help="Модель для сравнения: name config.yaml checkpoint.pth. Можно передать несколько раз.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/eval_d4", help="Куда сохранить CSV/JSON/PNG.")
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0 или cpu. По умолчанию из config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size для eval. По умолчанию из config.")
    parser.add_argument("--val_size", type=float, default=None, help="Переопределить validation split из config.")
    parser.add_argument("--seed", type=int, default=None, help="Переопределить random_seed из config.")
    parser.add_argument("--max_images_per_domain", type=int, default=None, help="Ограничить число val-картинок на домен.")
    parser.add_argument(
        "--allowed_elements",
        nargs="*",
        default=None,
        help="Какие D4 элементы проверять. По умолчанию все 8.",
    )
    parser.add_argument(
        "--negative_pairs_per_image",
        type=int,
        default=1,
        help="Сколько deterministic negative/null пар добавить на каждую val-картинку.",
    )
    parser.add_argument("--max_gen_len", type=int, default=None, help="Максимум генерируемых токенов после BOS.")
    parser.add_argument(
        "--continuation_max_prefix",
        type=int,
        default=4,
        help="Максимальная длина prefix для эксперимента продолжения последовательности.",
    )
    parser.add_argument("--save_predictions", action="store_true", help="Сохранить подробные prediction rows.")
    parser.add_argument("--no_plots", action="store_true", help="Не строить PNG-графики.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_model_specs(args.model)
    first_config = OmegaConf.load(specs[0].config_path)
    data_path = resolve_input_path(args.data_path)
    output_dir = resolve_output_path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    val_size = args.val_size if args.val_size is not None else float(first_config.training.get("val_size", 0.1))
    random_seed = args.seed if args.seed is not None else int(first_config.training.get("random_seed", 42))
    allowed_elements = args.allowed_elements or list(D4_CANONICAL_SEQUENCES.keys())

    val_items = collect_val_items(
        data_dir=data_path,
        val_size=val_size,
        random_seed=random_seed,
        max_images_per_domain=args.max_images_per_domain,
    )
    records = build_eval_records(
        val_items=val_items,
        allowed_elements=allowed_elements,
        negative_pairs_per_image=args.negative_pairs_per_image,
    )

    print("Оценка D4-моделей")
    print(f"  data_path: {data_path}")
    print(f"  val images: {len(val_items)}")
    print(f"  eval pairs: {len(records)}")
    print(f"  domains: {', '.join(sorted({item.domain for item in val_items}))}")
    print(f"  output_dir: {output_dir}")

    all_overall_rows: List[Dict[str, object]] = []
    all_domain_rows: List[Dict[str, object]] = []
    all_class_rows: List[Dict[str, object]] = []
    all_continuation_overall_rows: List[Dict[str, object]] = []
    all_continuation_domain_rows: List[Dict[str, object]] = []
    model_confusions: Dict[str, Counter] = {}

    for spec in specs:
        result = evaluate_model(spec, records, output_dir, args)
        all_overall_rows.append(result["overall_row"])
        all_domain_rows.extend(result["domain_rows"])
        all_class_rows.extend(result["class_rows"])
        all_continuation_overall_rows.extend(result["continuation_overall_rows"])
        all_continuation_domain_rows.extend(result["continuation_domain_rows"])
        model_confusions[spec.name] = result["confusion"]

    summary_payload = {
        "data_path": data_path,
        "val_images": len(val_items),
        "eval_pairs": len(records),
        "allowed_elements": list(allowed_elements),
        "negative_pairs_per_image": args.negative_pairs_per_image,
        "models": [spec.name for spec in specs],
        "overall": all_overall_rows,
    }

    write_json(str(Path(output_dir) / "metrics_summary.json"), summary_payload)
    write_csv(str(Path(output_dir) / "metrics_overall.csv"), all_overall_rows)
    write_csv(str(Path(output_dir) / "metrics_by_domain.csv"), all_domain_rows)
    write_csv(str(Path(output_dir) / "metrics_by_class.csv"), all_class_rows)
    write_csv(str(Path(output_dir) / "continuation_by_prefix.csv"), all_continuation_overall_rows)
    write_csv(str(Path(output_dir) / "continuation_by_domain_prefix.csv"), all_continuation_domain_rows)

    if not args.no_plots:
        build_plots(
            output_dir=output_dir,
            model_names=[spec.name for spec in specs],
            domain_rows=all_domain_rows,
            class_rows=all_class_rows,
            continuation_overall_rows=all_continuation_overall_rows,
            model_confusions=model_confusions,
        )

    print("\nФайлы сохранены:")
    print(f"  {Path(output_dir) / 'metrics_summary.json'}")
    print(f"  {Path(output_dir) / 'metrics_overall.csv'}")
    print(f"  {Path(output_dir) / 'metrics_by_domain.csv'}")
    print(f"  {Path(output_dir) / 'metrics_by_class.csv'}")
    print(f"  {Path(output_dir) / 'continuation_by_prefix.csv'}")
    print(f"  {Path(output_dir) / 'continuation_by_domain_prefix.csv'}")
    if not args.no_plots:
        print(f"  {Path(output_dir) / 'plots'}")


if __name__ == "__main__":
    main()
