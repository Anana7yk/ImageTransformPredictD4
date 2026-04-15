import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from src.dataset.augmentation import canonicalize_d4_sequence, inverse_d4_sequence
from src.dataset.tokenizer import (
    D4_CANONICAL_SEQUENCES,
    END_TOKEN_ID,
    ID_TO_TOKEN,
    NULL_SEQUENCE,
    PAD_TOKEN_ID,
    START_TOKEN_ID,
    TransformTokenizer,
)
from src.model import ImageTransformPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NULL_LABEL = "∅"
INVALID_LABEL = "invalid"
NO_TARGET_LABEL = "[NO_TARGET]"
NO_PRED_LABEL = "[NO_PRED]"
TOKEN_CLASS_LABELS = ["[END]", "e", "r", "s", "[NULL]", INVALID_LABEL, NO_TARGET_LABEL, NO_PRED_LABEL]
ELEMENT_CLASS_LABELS = list(D4_CANONICAL_SEQUENCES.keys()) + [NULL_LABEL, INVALID_LABEL]
SEQUENCE_TO_ELEMENT = {tuple(tokens): name for name, tokens in D4_CANONICAL_SEQUENCES.items()}


@dataclass
class ImageItem:
    path: str
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRecord:
    sample_id: str
    image1_path: str
    domain: str
    applied_element: str
    applied_tokens: List[str]
    target_element: str
    target_tokens: List[str]
    is_negative: bool = False
    image2_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImagePairBatch:
    image1: torch.Tensor
    image2: torch.Tensor
    target_ids: torch.Tensor
    records: List[EvalRecord]


@dataclass
class GeneratedBatch:
    ids: torch.Tensor
    mean_confidences: List[Optional[float]]
    mean_logprobs: List[Optional[float]]


def resolve_path(path_value: Optional[str], project_root: Path = PROJECT_ROOT) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return str(project_root / path)


def resolve_output_dir(path_value: str, project_root: Path = PROJECT_ROOT) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    if path.parent != Path(".") and (project_root / path.parent).exists():
        return str(project_root / path)
    return str(path.resolve())


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    finite_values = [value for value in values if value is not None and not math.isnan(value)]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def format_tokens(tokens: Optional[Sequence[str]]) -> str:
    if tokens is None:
        return "-"
    if not tokens:
        return "(empty)"
    return " ".join(str(token) for token in tokens)


def parse_token_string(raw_value: str) -> List[str]:
    value = raw_value.strip()
    if not value or value == "e":
        return ["e"]
    if value in {NULL_LABEL, "null", "NULL", "[NULL]"}:
        return list(NULL_SEQUENCE)
    return value.replace(",", " ").split()


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
    transformed = image.copy()
    for token in tokens:
        transformed = apply_token_to_image(transformed, token)
    return transformed.convert("RGB")


def canonical_element_from_tokens(tokens: Sequence[str]) -> Tuple[str, List[str]]:
    canonical_tokens = canonicalize_d4_sequence(list(tokens))
    return SEQUENCE_TO_ELEMENT[tuple(canonical_tokens)], canonical_tokens


def target_tokens_from_applied(applied_tokens: Sequence[str], target_mode: str) -> List[str]:
    if target_mode == "applied":
        return list(applied_tokens)
    if target_mode in {"completion", "inverse"}:
        return inverse_d4_sequence(list(applied_tokens))
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def decode_d4_label(tokens: Sequence[str]) -> Dict[str, Any]:
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
        "is_canonical": list(tokens) == canonical_tokens,
    }


def implied_applied_label(decoded: Dict[str, Any], target_mode: str) -> str:
    if decoded["label"] in {NULL_LABEL, INVALID_LABEL}:
        return str(decoded["label"])
    canonical_tokens = decoded.get("canonical_tokens")
    if not isinstance(canonical_tokens, list):
        return INVALID_LABEL
    if target_mode == "applied":
        return str(decoded["label"])
    applied_tokens = inverse_d4_sequence(canonical_tokens)
    applied_label, _ = canonical_element_from_tokens(applied_tokens)
    return applied_label


def normalize_split(split: str) -> str:
    split = split.lower()
    if split in {"heldout", "holdout", "validation", "valid", "test"}:
        return "val"
    if split in {"train", "val"}:
        return split
    raise ValueError("split must be one of: train, val, validation, heldout, test")


def collect_image_items(
    data_dir: str,
    split: str,
    val_size: float,
    random_seed: int,
    max_images_per_domain: Optional[int] = None,
) -> List[ImageItem]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    normalized_split = normalize_split(split)
    domain_to_paths: Dict[str, List[str]] = {}

    direct_images = [
        str(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
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
    selected_items: List[ImageItem] = []
    for domain, paths in domain_to_paths.items():
        shuffled = rng.sample(paths, len(paths))
        n_val = int(len(shuffled) * val_size)
        selected = shuffled[:n_val] if normalized_split == "val" else shuffled[n_val:]
        if max_images_per_domain is not None:
            selected = selected[:max_images_per_domain]
        selected_items.extend(ImageItem(path=path, domain=domain) for path in selected)

    if not selected_items:
        raise ValueError(
            "Selected split is empty. Check --data_path, --split, --val_size, "
            "or --max_images_per_domain."
        )
    return selected_items


def build_d4_eval_records(
    image_items: Sequence[ImageItem],
    allowed_elements: Sequence[str],
    target_mode: str,
    negative_pairs_per_image: int = 0,
) -> List[EvalRecord]:
    invalid = [element for element in allowed_elements if element not in D4_CANONICAL_SEQUENCES]
    if invalid:
        raise ValueError(f"Unknown D4 elements: {invalid}")

    records: List[EvalRecord] = []
    for item_index, item in enumerate(image_items):
        for element in allowed_elements:
            applied_tokens = list(D4_CANONICAL_SEQUENCES[element])
            target_tokens = target_tokens_from_applied(applied_tokens, target_mode)
            target_label = canonical_element_from_tokens(target_tokens)[0]
            records.append(
                EvalRecord(
                    sample_id=f"{item_index}:{element}",
                    image1_path=item.path,
                    domain=item.domain,
                    applied_element=element,
                    applied_tokens=applied_tokens,
                    target_element=target_label,
                    target_tokens=target_tokens,
                    metadata=dict(item.metadata),
                )
            )

    if negative_pairs_per_image > 0 and len(image_items) > 1:
        total_items = len(image_items)
        for item_index, item in enumerate(image_items):
            for neg_index in range(negative_pairs_per_image):
                other = image_items[(item_index + neg_index + 1) % total_items]
                if other.path == item.path:
                    other = image_items[(item_index + neg_index + 2) % total_items]
                records.append(
                    EvalRecord(
                        sample_id=f"{item_index}:null:{neg_index}",
                        image1_path=item.path,
                        image2_path=other.path,
                        domain=item.domain,
                        applied_element=NULL_LABEL,
                        applied_tokens=list(NULL_SEQUENCE),
                        target_element=NULL_LABEL,
                        target_tokens=list(NULL_SEQUENCE),
                        is_negative=True,
                        metadata={**item.metadata, "negative_domain": other.domain},
                    )
                )
    return records


def chunks(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def make_image_pair_batch(
    records: Sequence[EvalRecord],
    preprocess: Callable[[Image.Image], torch.Tensor],
    tokenizer: TransformTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> ImagePairBatch:
    image1_tensors = []
    image2_tensors = []
    target_ids = []

    for record in records:
        image1 = load_image(record.image1_path)
        if record.is_negative:
            if record.image2_path is None:
                raise ValueError("Negative EvalRecord must contain image2_path.")
            image2 = load_image(record.image2_path)
        else:
            image2 = apply_sequence_to_image(image1, record.applied_tokens)

        image1_tensors.append(preprocess(image1))
        image2_tensors.append(preprocess(image2))
        target_ids.append(
            tokenizer.encode(
                transforms=record.target_tokens,
                add_special_tokens=True,
                max_seq_len=max_seq_len,
                return_targets=False,
            )
        )

    return ImagePairBatch(
        image1=torch.stack(image1_tensors).to(device),
        image2=torch.stack(image2_tensors).to(device),
        target_ids=torch.stack(target_ids).to(device),
        records=list(records),
    )


def choose_device(device_arg: Optional[str], config: Any) -> torch.device:
    requested = device_arg or config.training.get("device", "cpu")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA недоступна, переключаюсь на CPU.")
        return torch.device("cpu")
    return torch.device(str(requested))


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
            "Для efficientnet checkpoint используй efficientnet config, "
            "для vit checkpoint используй vit config."
        ) from exc


def shifted_targets(input_ids: torch.Tensor, pad_token_id: int = PAD_TOKEN_ID) -> torch.Tensor:
    targets = torch.full_like(input_ids, pad_token_id)
    targets[:, :-1] = input_ids[:, 1:]
    return targets


def trim_at_pad(ids: Sequence[int]) -> List[int]:
    trimmed = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == PAD_TOKEN_ID:
            break
        trimmed.append(token_id)
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


def core_ids_from_input_ids(ids: Sequence[int]) -> List[int]:
    core = []
    for token_id in trim_at_pad(ids):
        token_id = int(token_id)
        if token_id == START_TOKEN_ID:
            continue
        if token_id == END_TOKEN_ID:
            break
        core.append(token_id)
    return core


def eval_token_ids_from_full_ids(ids: Sequence[int], generated: bool) -> List[int]:
    trimmed = trim_generated_ids(ids) if generated else trim_at_pad(ids)
    eval_ids = []
    for token_id in trimmed:
        token_id = int(token_id)
        if token_id in {START_TOKEN_ID, PAD_TOKEN_ID}:
            continue
        eval_ids.append(token_id)
        if token_id == END_TOKEN_ID:
            break
    return eval_ids


def ids_to_token_labels(ids: Sequence[int]) -> List[str]:
    return [ID_TO_TOKEN.get(int(token_id), "[UNK]") for token_id in ids]


def decoded_content_tokens(
    tokenizer: TransformTokenizer,
    ids: Sequence[int],
    generated: bool = True,
) -> List[str]:
    trimmed = trim_generated_ids(ids) if generated else trim_at_pad(ids)
    return tokenizer.decode(trimmed, skip_special_tokens=True)


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            substitution = prev[j - 1] + int(left_item != right_item)
            insertion = current[j - 1] + 1
            deletion = prev[j] + 1
            current.append(min(substitution, insertion, deletion))
        prev = current
    return prev[-1]


def sequence_comparison(
    tokenizer: TransformTokenizer,
    generated_ids: Sequence[int],
    target_ids: Sequence[int],
) -> Dict[str, Any]:
    generated_trimmed = trim_generated_ids(generated_ids)
    target_trimmed = trim_at_pad(target_ids)
    exact_with_eos = generated_trimmed == target_trimmed

    pred_eval_ids = eval_token_ids_from_full_ids(generated_ids, generated=True)
    target_eval_ids = eval_token_ids_from_full_ids(target_ids, generated=False)
    max_len = max(len(pred_eval_ids), len(target_eval_ids))
    token_correct = 0
    for index in range(max_len):
        pred_id = pred_eval_ids[index] if index < len(pred_eval_ids) else None
        target_id = target_eval_ids[index] if index < len(target_eval_ids) else None
        token_correct += int(pred_id == target_id)

    pred_content = decoded_content_tokens(tokenizer, generated_ids, generated=True)
    target_content = decoded_content_tokens(tokenizer, target_ids, generated=False)
    distance = edit_distance(pred_content, target_content)
    norm_distance = safe_div(distance, max(len(pred_content), len(target_content), 1))

    return {
        "exact_match": exact_with_eos,
        "content_exact_match": pred_content == target_content,
        "token_correct": token_correct,
        "token_total": max_len,
        "token_accuracy": safe_div(token_correct, max_len),
        "edit_distance": distance,
        "normalized_edit_distance": norm_distance,
        "pred_eval_token_labels": ids_to_token_labels(pred_eval_ids),
        "target_eval_token_labels": ids_to_token_labels(target_eval_ids),
        "pred_content_tokens": pred_content,
        "target_content_tokens": target_content,
    }


def aligned_token_pairs(
    target_labels: Sequence[str],
    pred_labels: Sequence[str],
) -> List[Tuple[str, str]]:
    max_len = max(len(target_labels), len(pred_labels))
    pairs = []
    for index in range(max_len):
        target = target_labels[index] if index < len(target_labels) else NO_TARGET_LABEL
        pred = pred_labels[index] if index < len(pred_labels) else NO_PRED_LABEL
        pairs.append((target, pred))
    return pairs


@torch.no_grad()
def greedy_decode_from_embeddings(
    model: ImageTransformPredictor,
    images_embeddings: torch.Tensor,
    max_new_tokens: int,
    prefix_ids: Optional[torch.Tensor] = None,
) -> GeneratedBatch:
    batch_size = images_embeddings.shape[0]
    device = images_embeddings.device
    if prefix_ids is None:
        idx = torch.full((batch_size, 1), model.bos_token_id, dtype=torch.long, device=device)
    else:
        idx = prefix_ids.to(device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    confidence_values: List[List[float]] = [[] for _ in range(batch_size)]
    logprob_values: List[List[float]] = [[] for _ in range(batch_size)]

    for _ in range(max_new_tokens):
        logits, _ = model.transform_decoder(idx=idx, images_embeddings=images_embeddings, targets=None)
        next_logits = logits[:, -1, :].clone()
        next_logits[:, PAD_TOKEN_ID] = -float("inf")
        next_logits[:, START_TOKEN_ID] = -float("inf")

        log_probs = F.log_softmax(next_logits, dim=-1)
        probs = log_probs.exp()
        next_ids = probs.argmax(dim=-1)
        next_probs = probs.gather(1, next_ids.unsqueeze(1)).squeeze(1)
        next_logprobs = log_probs.gather(1, next_ids.unsqueeze(1)).squeeze(1)

        active = ~finished
        for sample_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            confidence_values[sample_index].append(float(next_probs[sample_index].item()))
            logprob_values[sample_index].append(float(next_logprobs[sample_index].item()))

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
    mean_logprobs: List[Optional[float]] = []
    for probs_for_sample, logprobs_for_sample in zip(confidence_values, logprob_values):
        mean_confidences.append(sum(probs_for_sample) / len(probs_for_sample) if probs_for_sample else None)
        mean_logprobs.append(sum(logprobs_for_sample) / len(logprobs_for_sample) if logprobs_for_sample else None)

    return GeneratedBatch(
        ids=idx[:, :target_len],
        mean_confidences=mean_confidences,
        mean_logprobs=mean_logprobs,
    )


def compute_teacher_forced_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int = PAD_TOKEN_ID,
) -> Dict[str, Any]:
    targets = shifted_targets(input_ids, pad_token_id)
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

    losses = []
    confidences = []
    for sample_index in range(input_ids.shape[0]):
        sample_mask = mask[sample_index]
        if int(sample_mask.sum().item()) == 0:
            losses.append(0.0)
            confidences.append(None)
            continue
        losses.append(float(token_losses[sample_index][sample_mask].mean().item()))
        confidences.append(float(target_probs[sample_index][sample_mask].mean().item()))

    return {
        "targets": targets,
        "seq_correct": seq_correct.detach().cpu().tolist(),
        "token_correct": correct.sum(dim=1).detach().cpu().tolist(),
        "token_total": mask.sum(dim=1).detach().cpu().tolist(),
        "loss": losses,
        "target_confidence": confidences,
    }


def classification_summary(
    confusion: Counter,
    labels: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    true_counts = Counter()
    pred_counts = Counter()
    for (true_label, pred_label), count in confusion.items():
        true_counts[true_label] += count
        pred_counts[pred_label] += count

    rows = []
    macro_precision = []
    macro_recall = []
    macro_f1 = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    total_support = sum(true_counts[label] for label in labels)
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for label in labels:
        tp = confusion[(label, label)]
        fp = pred_counts[label] - tp
        fn = true_counts[label] - tp
        support = true_counts[label]
        predicted = pred_counts[label]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        tp_total += tp
        fp_total += fp
        fn_total += fn
        rows.append(
            {
                "class": label,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    micro_precision = safe_div(tp_total, tp_total + fp_total)
    micro_recall = safe_div(tp_total, tp_total + fn_total)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return (
        {
            "precision_micro": micro_precision,
            "recall_micro": micro_recall,
            "f1_micro": micro_f1,
            "precision_macro": safe_div(sum(macro_precision), len(macro_precision)),
            "recall_macro": safe_div(sum(macro_recall), len(macro_recall)),
            "f1_macro": safe_div(sum(macro_f1), len(macro_f1)),
            "precision_weighted": safe_div(weighted_precision, total_support),
            "recall_weighted": safe_div(weighted_recall, total_support),
            "f1_weighted": safe_div(weighted_f1, total_support),
        },
        rows,
    )


def _row_bool(row: Dict[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "да"}
    return bool(value)


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
    exact_key: str,
    class_labels: Sequence[str] = ELEMENT_CLASS_LABELS,
    token_labels: Sequence[str] = TOKEN_CLASS_LABELS,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Counter, Counter]:
    domain_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domain_to_rows[str(row.get("domain", "overall"))].append(row)

    overall_metrics, class_rows, token_rows, class_confusion, token_confusion = aggregate_subset(
        rows=rows,
        domain="overall",
        exact_key=exact_key,
        class_labels=class_labels,
        token_labels=token_labels,
    )

    domain_metrics = []
    all_class_rows = list(class_rows)
    all_token_rows = list(token_rows)
    for domain, domain_rows in sorted(domain_to_rows.items()):
        metrics, domain_class_rows, domain_token_rows, _, _ = aggregate_subset(
            rows=domain_rows,
            domain=domain,
            exact_key=exact_key,
            class_labels=class_labels,
            token_labels=token_labels,
        )
        domain_metrics.append(metrics)
        all_class_rows.extend(domain_class_rows)
        all_token_rows.extend(domain_token_rows)

    return overall_metrics, domain_metrics, all_class_rows, all_token_rows, class_confusion, token_confusion


def aggregate_subset(
    rows: Sequence[Dict[str, Any]],
    domain: str,
    exact_key: str,
    class_labels: Sequence[str],
    token_labels: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]], Counter, Counter]:
    class_confusion: Counter = Counter()
    token_confusion: Counter = Counter()

    count = len(rows)
    exact_count = 0
    content_exact_count = 0
    next_token_count = 0
    token_correct = 0
    token_total = 0
    edit_sum = 0.0
    normalized_edit_sum = 0.0
    target_len_sum = 0.0
    pred_len_sum = 0.0
    confidence_values: List[Optional[float]] = []
    confidence_exact_values: List[Optional[float]] = []
    confidence_error_values: List[Optional[float]] = []
    logprob_values: List[Optional[float]] = []
    loss_values: List[Optional[float]] = []
    teacher_conf_values: List[Optional[float]] = []

    for row in rows:
        exact = _row_bool(row, exact_key)
        content_exact = _row_bool(row, "content_exact_match")
        next_correct = _row_bool(row, "next_token_correct")
        exact_count += int(exact)
        content_exact_count += int(content_exact)
        next_token_count += int(next_correct)
        token_correct += int(row.get("token_correct", 0))
        token_total += int(row.get("token_total", 0))
        edit_sum += float(row.get("edit_distance", 0.0))
        normalized_edit_sum += float(row.get("normalized_edit_distance", 0.0))
        target_len_sum += float(row.get("target_length", row.get("target_continuation_length", 0)))
        pred_len_sum += float(row.get("prediction_length", row.get("predicted_continuation_length", 0)))

        confidence = row.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
        confidence_values.append(confidence)
        if exact:
            confidence_exact_values.append(confidence)
        else:
            confidence_error_values.append(confidence)

        logprob = row.get("mean_logprob")
        if logprob is not None:
            logprob_values.append(float(logprob))

        loss = row.get("teacher_loss")
        if loss is not None:
            loss_values.append(float(loss))

        teacher_conf = row.get("teacher_target_confidence")
        if teacher_conf is not None:
            teacher_conf_values.append(float(teacher_conf))

        class_confusion[(str(row.get("target_label", INVALID_LABEL)), str(row.get("predicted_label", INVALID_LABEL)))] += 1
        for target_token, pred_token in aligned_token_pairs(
            row.get("target_token_labels", []),
            row.get("pred_token_labels", []),
        ):
            token_confusion[(target_token, pred_token)] += 1

    class_summary, class_rows = classification_summary(class_confusion, class_labels)
    token_summary, token_rows = classification_summary(token_confusion, token_labels)
    for class_row in class_rows:
        class_row["domain"] = domain
    for token_row in token_rows:
        token_row["domain"] = domain

    metrics = {
        "domain": domain,
        "samples": count,
        "sequence_accuracy": safe_div(exact_count, count),
        "exact_match": safe_div(exact_count, count),
        "content_exact_match": safe_div(content_exact_count, count),
        "next_token_accuracy": safe_div(next_token_count, count),
        "token_accuracy": safe_div(token_correct, token_total),
        "avg_edit_distance": safe_div(edit_sum, count),
        "avg_normalized_edit_distance": safe_div(normalized_edit_sum, count),
        "avg_target_length": safe_div(target_len_sum, count),
        "avg_prediction_length": safe_div(pred_len_sum, count),
        "avg_confidence": mean_or_none(confidence_values) or 0.0,
        "avg_confidence_exact": mean_or_none(confidence_exact_values) or 0.0,
        "avg_confidence_errors": mean_or_none(confidence_error_values) or 0.0,
        "avg_mean_logprob": mean_or_none(logprob_values) or 0.0,
        "avg_teacher_loss": mean_or_none(loss_values) or 0.0,
        "avg_teacher_target_confidence": mean_or_none(teacher_conf_values) or 0.0,
    }
    metrics.update({f"class_{key}": value for key, value in class_summary.items()})
    metrics.update({f"token_{key}": value for key, value in token_summary.items()})
    return metrics, class_rows, token_rows, class_confusion, token_confusion


def rows_by_integer_field(
    rows: Sequence[Dict[str, Any]],
    field_name: str,
    exact_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get(field_name, 0))].append(row)

    result = []
    for field_value, items in sorted(grouped.items()):
        exact = sum(int(_row_bool(item, exact_key)) for item in items)
        token_correct = sum(int(item.get("token_correct", 0)) for item in items)
        token_total = sum(int(item.get("token_total", 0)) for item in items)
        confidences = [item.get("confidence") for item in items]
        result.append(
            {
                field_name: field_value,
                "samples": len(items),
                "exact_match": safe_div(exact, len(items)),
                "token_accuracy": safe_div(token_correct, token_total),
                "avg_confidence": mean_or_none([float(value) if value is not None else None for value in confidences]) or 0.0,
            }
        )
    return result


def rows_to_csv_ready(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    csv_rows = []
    for row in rows:
        csv_row = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple)):
                csv_row[key] = format_tokens(value)
            elif isinstance(value, dict):
                csv_row[key] = json.dumps(value, ensure_ascii=False)
            else:
                csv_row[key] = value
        csv_rows.append(csv_row)
    return csv_rows


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    csv_rows = rows_to_csv_ready(rows)
    fieldnames = list(csv_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_md(
    path: str,
    title: str,
    config_payload: Dict[str, Any],
    overall_metrics: Dict[str, Any],
    metric_notes: Sequence[str],
) -> None:
    lines = [f"# {title}", ""]
    lines.append("## Run")
    for key, value in config_payload.items():
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")
    lines.append("## Main Metrics")
    for key in [
        "samples",
        "sequence_accuracy",
        "token_accuracy",
        "class_precision_macro",
        "class_recall_macro",
        "class_f1_macro",
        "token_precision_macro",
        "token_recall_macro",
        "token_f1_macro",
        "avg_confidence",
        "avg_normalized_edit_distance",
    ]:
        if key in overall_metrics:
            value = overall_metrics[key]
            lines.append(f"- **{key}**: {value:.6f}" if isinstance(value, float) else f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Metric Notes")
    for note in metric_notes:
        lines.append(f"- {note}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_get_plt():
    try:
        import matplotlib.pyplot as plt

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            pass
        return plt
    except Exception as exc:
        print(f"matplotlib недоступен, графики не будут построены: {exc}")
        return None


def save_plot(fig: Any, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")


def plot_domain_metrics(
    output_dir: str,
    domain_rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    title: str,
) -> None:
    plt = _maybe_get_plt()
    if plt is None:
        return
    domains = [row["domain"] for row in domain_rows if row["domain"] != "overall"]
    if not domains:
        return
    x_positions = list(range(len(domains)))
    width = 0.8 / max(1, len(metrics))
    colors = ["#2D5BFF", "#E4572E", "#17A398", "#F2AF29", "#7A4EAB"]

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 1.25), 5.8))
    for metric_index, metric in enumerate(metrics):
        values = [float(row.get(metric, 0.0)) for row in domain_rows if row["domain"] != "overall"]
        offsets = [x + (metric_index - (len(metrics) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=metric, color=colors[metric_index % len(colors)], alpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Значение")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(domains, rotation=35, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, Path(output_dir) / "plots" / "quality_by_domain")
    plt.close(fig)


def plot_confidence_by_domain(output_dir: str, domain_rows: Sequence[Dict[str, Any]]) -> None:
    plt = _maybe_get_plt()
    if plt is None:
        return
    rows = [row for row in domain_rows if row["domain"] != "overall"]
    if not rows:
        return
    domains = [row["domain"] for row in rows]
    values = [float(row.get("avg_confidence", 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(max(9, len(domains) * 1.2), 5.2))
    ax.bar(domains, values, color="#17A398", alpha=0.9)
    ax.set_title("Средняя уверенность по доменам", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average confidence")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, Path(output_dir) / "plots" / "confidence_by_domain")
    plt.close(fig)


def plot_histogram(
    output_dir: str,
    values: Sequence[float],
    output_name: str,
    title: str,
    xlabel: str,
    bins: int = 24,
) -> None:
    plt = _maybe_get_plt()
    if plt is None or not values:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(values, bins=bins, color="#2D5BFF", alpha=0.82, edgecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Количество примеров")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, Path(output_dir) / "plots" / output_name)
    plt.close(fig)


def plot_exact_match_counts(output_dir: str, rows: Sequence[Dict[str, Any]], exact_key: str) -> None:
    plt = _maybe_get_plt()
    if plt is None:
        return
    exact = sum(int(_row_bool(row, exact_key)) for row in rows)
    errors = len(rows) - exact
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.bar(["exact match", "error"], [exact, errors], color=["#17A398", "#E4572E"], alpha=0.9)
    ax.set_title("Exact match / non-exact match", fontsize=14, fontweight="bold")
    ax.set_ylabel("Количество примеров")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, Path(output_dir) / "plots" / "exact_match_counts")
    plt.close(fig)


def plot_confidence_by_correctness(output_dir: str, rows: Sequence[Dict[str, Any]], exact_key: str) -> None:
    plt = _maybe_get_plt()
    if plt is None:
        return
    exact_conf = [float(row["confidence"]) for row in rows if _row_bool(row, exact_key) and row.get("confidence") is not None]
    error_conf = [float(row["confidence"]) for row in rows if not _row_bool(row, exact_key) and row.get("confidence") is not None]
    if not exact_conf and not error_conf:
        return
    data = []
    labels = []
    if exact_conf:
        data.append(exact_conf)
        labels.append("exact")
    if error_conf:
        data.append(error_conf)
        labels.append("non-exact")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_title("Confidence vs correctness", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean chosen-token probability")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, Path(output_dir) / "plots" / "confidence_vs_correctness")
    plt.close(fig)


def plot_metric_by_integer_field(
    output_dir: str,
    rows: Sequence[Dict[str, Any]],
    field_name: str,
    exact_key: str,
    output_name: str,
    title: str,
) -> None:
    plt = _maybe_get_plt()
    if plt is None:
        return
    summary = rows_by_integer_field(rows, field_name, exact_key)
    if not summary:
        return
    x_values = [int(row[field_name]) for row in summary]
    exact_values = [float(row["exact_match"]) for row in summary]
    token_values = [float(row["token_accuracy"]) for row in summary]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(x_values, exact_values, marker="o", linewidth=2.4, label="Exact match")
    ax.plot(x_values, token_values, marker="s", linewidth=2.0, linestyle="--", label="Token accuracy")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(field_name)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_plot(fig, Path(output_dir) / "plots" / output_name)
    plt.close(fig)


def plot_token_confusion(
    output_dir: str,
    token_confusion: Counter,
    labels: Sequence[str] = TOKEN_CLASS_LABELS,
) -> None:
    plt = _maybe_get_plt()
    if plt is None:
        return
    active_labels = [
        label for label in labels
        if any(token_confusion[(label, pred)] > 0 or token_confusion[(true, label)] > 0 for pred in labels for true in labels)
    ]
    if not active_labels:
        active_labels = list(labels)
    matrix = [[token_confusion[(target, pred)] for pred in active_labels] for target in active_labels]

    fig, ax = plt.subplots(figsize=(max(6.5, len(active_labels) * 0.8), max(5.8, len(active_labels) * 0.65)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Token confusion summary", fontsize=14, fontweight="bold")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_xticks(range(len(active_labels)))
    ax.set_xticklabels(active_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(active_labels)))
    ax.set_yticklabels(active_labels)
    max_value = max([max(row) for row in matrix] or [0])
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if max_value and value > max_value * 0.55 else "#111111"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_plot(fig, Path(output_dir) / "plots" / "token_confusion")
    plt.close(fig)
