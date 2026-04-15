#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.augmentation import inverse_d4_sequence
from src.dataset.tokenizer import D4_CANONICAL_SEQUENCES, PAD_TOKEN_ID, TransformTokenizer
from src.evaluation_utils import (
    ELEMENT_CLASS_LABELS,
    INVALID_LABEL,
    NULL_LABEL,
    aggregate_rows,
    build_d4_eval_records,
    choose_device,
    chunks,
    collect_image_items,
    compute_teacher_forced_metrics,
    decode_d4_label,
    edit_distance,
    greedy_decode_from_embeddings,
    load_model_checkpoint,
    make_image_pair_batch,
    plot_confidence_by_correctness,
    plot_confidence_by_domain,
    plot_domain_metrics,
    plot_exact_match_counts,
    plot_histogram,
    plot_token_confusion,
    resolve_output_dir,
    resolve_path,
    safe_div,
    sequence_comparison,
    shifted_targets,
    write_csv,
    write_json,
    write_jsonl,
    write_summary_md,
)
from src.model import ImageTransformPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an EfficientNet encoder + autoregressive decoder D4 sequence model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--data_path", type=str, default="data", help="Dataset root with domain subfolders.")
    parser.add_argument("--split", type=str, default="heldout", help="train, val, validation, heldout, or test.")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_efficient_sequence")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch_size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--val_size", type=float, default=None, help="Override config training.val_size.")
    parser.add_argument("--random_seed", type=int, default=None, help="Override config training.random_seed.")
    parser.add_argument("--max_images_per_domain", type=int, default=None, help="Limit held-out base images per domain.")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit final expanded image-pair examples.")
    parser.add_argument("--max_gen_len", type=int, default=None, help="Max autoregressive tokens after BOS.")
    parser.add_argument("--allowed_elements", nargs="*", default=None, help="D4 elements to evaluate. Default: all 8.")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="applied",
        choices=["applied", "completion", "inverse"],
        help=(
            "What the decoder was trained to output. Default is 'applied', matching the old "
            "augmentator: I2 = g(I1), target = g."
        ),
    )
    parser.add_argument(
        "--negative_pairs_per_image",
        type=int,
        default=0,
        help="Deterministic unrelated/null pairs per held-out image. Default is 0, matching the old augmentator.",
    )
    parser.add_argument(
        "--allow_non_efficient_encoder",
        action="store_true",
        help="Do not fail if config encoder is not efficientnet_encoder.",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--save_jsonl", action="store_true", help="Also save predictions.jsonl.")
    return parser.parse_args()


def default_negative_pairs(config: Any, explicit_value: Optional[int]) -> int:
    del config
    if explicit_value is not None:
        return explicit_value
    return 0


def model_output_to_transform(
    decoded_output: Dict[str, Any],
    target_mode: str,
) -> tuple[str, Optional[List[str]]]:
    """Convert raw decoder output to the actual transform g between I1 and I2."""
    output_label = str(decoded_output["label"])
    if output_label == NULL_LABEL:
        return NULL_LABEL, ["[NULL]"]
    if output_label == INVALID_LABEL:
        return INVALID_LABEL, None

    output_tokens = decoded_output.get("canonical_tokens")
    if not isinstance(output_tokens, list):
        return INVALID_LABEL, None

    if target_mode == "applied":
        transform_tokens = list(output_tokens)
    else:
        transform_tokens = inverse_d4_sequence(output_tokens)

    transform_decoded = decode_d4_label(transform_tokens)
    if transform_decoded["label"] == INVALID_LABEL:
        return INVALID_LABEL, None
    return str(transform_decoded["label"]), transform_decoded["canonical_tokens"]


def compare_transform_tokens(
    predicted_tokens: Optional[List[str]],
    target_tokens: List[str],
) -> Dict[str, Any]:
    predicted_for_metrics = predicted_tokens if predicted_tokens is not None else [INVALID_LABEL]
    target_labels = list(target_tokens) + ["[END]"]
    pred_labels = list(predicted_for_metrics) + ["[END]"]

    max_len = max(len(target_labels), len(pred_labels))
    correct = 0
    for index in range(max_len):
        target = target_labels[index] if index < len(target_labels) else None
        pred = pred_labels[index] if index < len(pred_labels) else None
        correct += int(target == pred)

    distance = edit_distance(predicted_for_metrics, target_tokens)
    return {
        "token_correct": correct,
        "token_total": max_len,
        "token_accuracy": safe_div(correct, max_len),
        "edit_distance": distance,
        "normalized_edit_distance": safe_div(distance, max(len(predicted_for_metrics), len(target_tokens), 1)),
        "target_token_labels": target_labels,
        "pred_token_labels": pred_labels,
    }


def build_prediction_rows(
    tokenizer: TransformTokenizer,
    batch,
    generated_ids: torch.Tensor,
    generated_confidences: List[Optional[float]],
    generated_logprobs: List[Optional[float]],
    teacher_metrics: Dict[str, Any],
    target_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    generated_cpu = generated_ids.detach().cpu().tolist()
    target_cpu = batch.target_ids.detach().cpu().tolist()

    for index, record in enumerate(batch.records):
        comparison = sequence_comparison(
            tokenizer=tokenizer,
            generated_ids=generated_cpu[index],
            target_ids=target_cpu[index],
        )
        decoded = decode_d4_label(comparison["pred_content_tokens"])
        model_output_label = str(decoded["label"])
        predicted_transform_label, predicted_transform_tokens = model_output_to_transform(decoded, target_mode)
        transform_comparison = compare_transform_tokens(
            predicted_tokens=predicted_transform_tokens,
            target_tokens=record.applied_tokens,
        )
        transform_exact = predicted_transform_label == record.applied_element

        teacher_token_total = int(teacher_metrics["token_total"][index])
        teacher_token_correct = int(teacher_metrics["token_correct"][index])
        rows.append(
            {
                "sample_id": record.sample_id,
                "domain": record.domain,
                "image1": record.image1_path,
                "image2": record.image2_path or f"<generated:{record.applied_element}>",
                "is_negative": record.is_negative,
                "applied_element": record.applied_element,
                "applied_tokens": record.applied_tokens,
                "target_label": record.applied_element,
                "target_tokens": record.applied_tokens,
                "predicted_label": predicted_transform_label,
                "predicted_tokens": predicted_transform_tokens,
                "exact_match": transform_exact,
                "content_exact_match": transform_exact,
                "token_accuracy": float(transform_comparison["token_accuracy"]),
                "token_correct": int(transform_comparison["token_correct"]),
                "token_total": int(transform_comparison["token_total"]),
                "edit_distance": int(transform_comparison["edit_distance"]),
                "normalized_edit_distance": float(transform_comparison["normalized_edit_distance"]),
                "target_length": len(record.applied_tokens),
                "prediction_length": len(predicted_transform_tokens or []),
                "target_token_labels": transform_comparison["target_token_labels"],
                "pred_token_labels": transform_comparison["pred_token_labels"],
                "model_output_target_label": record.target_element,
                "model_output_target_tokens": record.target_tokens,
                "model_output_predicted_label": model_output_label,
                "model_output_predicted_tokens_raw": comparison["pred_content_tokens"],
                "model_output_predicted_tokens_canonical": decoded.get("canonical_tokens"),
                "model_output_exact_match": bool(comparison["exact_match"]),
                "model_output_content_exact_match": bool(comparison["content_exact_match"]),
                "model_output_token_accuracy": float(comparison["token_accuracy"]),
                "model_output_token_correct": int(comparison["token_correct"]),
                "model_output_token_total": int(comparison["token_total"]),
                "model_output_edit_distance": int(comparison["edit_distance"]),
                "model_output_normalized_edit_distance": float(comparison["normalized_edit_distance"]),
                "teacher_sequence_correct": bool(teacher_metrics["seq_correct"][index]),
                "teacher_token_accuracy": teacher_token_correct / teacher_token_total if teacher_token_total else 0.0,
                "teacher_loss": float(teacher_metrics["loss"][index]),
                "teacher_target_confidence": teacher_metrics["target_confidence"][index],
                "confidence": generated_confidences[index],
                "mean_logprob": generated_logprobs[index],
                "generated_token_ids": generated_cpu[index],
                "target_token_ids": target_cpu[index],
                "is_canonical_prediction": bool(decoded.get("is_canonical", False)),
            }
        )
    return rows


def evaluate(args: argparse.Namespace) -> None:
    config_path = str(resolve_path(args.config))
    checkpoint_path = str(resolve_path(args.checkpoint))
    data_path = str(resolve_path(args.data_path))
    output_dir = resolve_output_dir(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)
    encoder_type = str(config.model.encoder.get("type", ""))
    if not args.allow_non_efficient_encoder and encoder_type not in {"efficientnet", "efficientnet_encoder"}:
        raise ValueError(
            f"This script is intended for efficientnet checkpoints, but config encoder.type={encoder_type!r}. "
            "Pass --allow_non_efficient_encoder if this is intentional."
        )

    device = choose_device(args.device, config)
    tokenizer = TransformTokenizer()
    model = ImageTransformPredictor(config.model)
    model.to(device)
    model.eval()
    load_model_checkpoint(model, checkpoint_path, device, config_path)

    batch_size = args.batch_size or int(config.training.get("batch_size", 8))
    val_size = args.val_size if args.val_size is not None else float(config.training.get("val_size", 0.1))
    random_seed = args.random_seed if args.random_seed is not None else int(config.training.get("random_seed", 42))
    max_seq_len = int(config.model.decoder.max_seq_len)
    max_gen_len = args.max_gen_len or (max_seq_len - 1)
    target_mode = args.target_mode
    allowed_elements = args.allowed_elements or list(config.data.get("allowed_d4_elements", D4_CANONICAL_SEQUENCES.keys()))
    negative_pairs = default_negative_pairs(config, args.negative_pairs_per_image)

    image_items = collect_image_items(
        data_dir=data_path,
        split=args.split,
        val_size=val_size,
        random_seed=random_seed,
        max_images_per_domain=args.max_images_per_domain,
    )
    records = build_d4_eval_records(
        image_items=image_items,
        allowed_elements=allowed_elements,
        target_mode=target_mode,
        negative_pairs_per_image=negative_pairs,
    )
    if args.max_examples is not None:
        records = records[:args.max_examples]
    if not records:
        raise RuntimeError("No evaluation records were built.")

    print("Efficient sequence evaluation")
    print(f"  encoder: {encoder_type}")
    print(f"  target_mode: {target_mode}")
    print("  old augmentator semantics: I2 = g(I1), target = g")
    print("  evaluated answer: D4 transform g between the two images")
    config_target_mode = config.data.get("target_mode", None)
    if config_target_mode is not None and config_target_mode != target_mode:
        print(f"  config data.target_mode={config_target_mode!r} is ignored here unless passed via --target_mode")
    if target_mode != "applied":
        print("  note: target_mode is not applied, raw decoder output will be converted to g = h^-1")
    print(f"  split: {args.split}")
    print(f"  held-out images: {len(image_items)}")
    print(f"  eval pairs: {len(records)}")
    print(f"  output_dir: {output_dir}")

    prediction_rows: List[Dict[str, Any]] = []
    preprocess = model.image_pair_encoder.preprocess

    with torch.no_grad():
        for record_batch in tqdm(list(chunks(records, batch_size)), desc="Evaluating sequence model"):
            batch = make_image_pair_batch(
                records=record_batch,
                preprocess=preprocess,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                device=device,
            )
            images_embeddings = model.image_pair_encoder(batch.image1, batch.image2)
            targets = shifted_targets(batch.target_ids, PAD_TOKEN_ID)
            logits, _ = model.transform_decoder(idx=batch.target_ids, images_embeddings=images_embeddings, targets=targets)
            teacher_metrics = compute_teacher_forced_metrics(logits, batch.target_ids, PAD_TOKEN_ID)
            generated = greedy_decode_from_embeddings(
                model=model,
                images_embeddings=images_embeddings,
                max_new_tokens=max_gen_len,
            )
            prediction_rows.extend(
                build_prediction_rows(
                    tokenizer=tokenizer,
                    batch=batch,
                    generated_ids=generated.ids,
                    generated_confidences=generated.mean_confidences,
                    generated_logprobs=generated.mean_logprobs,
                    teacher_metrics=teacher_metrics,
                    target_mode=target_mode,
                )
            )

    overall_metrics, domain_rows, class_rows, token_rows, _, token_confusion = aggregate_rows(
        prediction_rows,
        exact_key="exact_match",
        class_labels=ELEMENT_CLASS_LABELS,
    )

    run_config = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "data_path": data_path,
        "split": args.split,
        "target_mode": target_mode,
        "config_target_mode": config.data.get("target_mode", None),
        "encoder": encoder_type,
        "batch_size": batch_size,
        "max_gen_len": max_gen_len,
        "negative_pairs_per_image": negative_pairs,
        "heldout_images": len(image_items),
        "eval_pairs": len(records),
    }
    metric_notes = [
        "Primary metrics evaluate the old augmentator task: given two images, predict g such that I2 = g(I1).",
        "By default target_mode=applied, so the expected output is the exact canonical sequence returned by the old ImageTransformer.",
        "If target_mode=completion/inverse is explicitly passed, the raw decoder output h is converted to g=h^-1 before scoring.",
        "Sequence Accuracy / Exact Match is exact D4-element match for the predicted transform g.",
        "Token Accuracy compares canonical generator tokens for g position-wise; EOS and [NULL] are included, PAD/BOS are excluded.",
        "Precision/Recall/F1 are reported for D4 transform labels and token labels. Raw decoder output is preserved in model_output_* columns.",
    ]
    metrics_payload = {
        "run": run_config,
        "metric_notes": metric_notes,
        "overall": overall_metrics,
        "by_domain": domain_rows,
    }

    write_json(str(Path(output_dir) / "metrics.json"), metrics_payload)
    write_csv(str(Path(output_dir) / "metrics_by_domain.csv"), domain_rows)
    write_csv(str(Path(output_dir) / "metrics_by_class.csv"), class_rows)
    write_csv(str(Path(output_dir) / "metrics_by_token.csv"), token_rows)
    write_csv(str(Path(output_dir) / "predictions.csv"), prediction_rows)
    if args.save_jsonl:
        write_jsonl(str(Path(output_dir) / "predictions.jsonl"), prediction_rows)
    write_summary_md(
        path=str(Path(output_dir) / "summary.md"),
        title="Efficient Sequence Model Evaluation",
        config_payload=run_config,
        overall_metrics=overall_metrics,
        metric_notes=metric_notes,
    )

    if not args.no_plots:
        plot_domain_metrics(
            output_dir=output_dir,
            domain_rows=domain_rows,
            metrics=["sequence_accuracy", "token_accuracy", "class_f1_macro", "avg_confidence"],
            title="Качество определения D4-преобразования по доменам",
        )
        plot_confidence_by_domain(output_dir, domain_rows)
        plot_exact_match_counts(output_dir, prediction_rows, exact_key="exact_match")
        plot_confidence_by_correctness(output_dir, prediction_rows, exact_key="exact_match")
        plot_histogram(
            output_dir,
            [float(row["edit_distance"]) for row in prediction_rows],
            output_name="edit_distance_hist",
            title="Edit distance для предсказанного D4-преобразования",
            xlabel="Edit distance по canonical g-токенам",
        )
        plot_histogram(
            output_dir,
            [float(row["normalized_edit_distance"]) for row in prediction_rows],
            output_name="normalized_edit_distance_hist",
            title="Normalized edit distance для D4-преобразования",
            xlabel="Normalized edit distance по canonical g-токенам",
        )
        plot_token_confusion(output_dir, token_confusion)

    print("\nSaved:")
    print(f"  {Path(output_dir) / 'metrics.json'}")
    print(f"  {Path(output_dir) / 'metrics_by_domain.csv'}")
    print(f"  {Path(output_dir) / 'predictions.csv'}")
    print(f"  {Path(output_dir) / 'summary.md'}")
    if not args.no_plots:
        print(f"  {Path(output_dir) / 'plots'}")
    print("\nMain metrics:")
    print(f"  transform_sequence_accuracy: {overall_metrics['sequence_accuracy']:.4f}")
    print(f"  transform_token_accuracy: {overall_metrics['token_accuracy']:.4f}")
    print(f"  transform_class_f1_macro: {overall_metrics['class_f1_macro']:.4f}")
    print(f"  transform_token_f1_macro: {overall_metrics['token_f1_macro']:.4f}")
    print(f"  avg_confidence: {overall_metrics['avg_confidence']:.4f}")


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
