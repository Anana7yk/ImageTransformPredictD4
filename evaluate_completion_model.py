#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.tokenizer import D4_CANONICAL_SEQUENCES, END_TOKEN_ID, PAD_TOKEN_ID, START_TOKEN_ID, TransformTokenizer
from src.evaluation_utils import (
    ELEMENT_CLASS_LABELS,
    aggregate_rows,
    build_d4_eval_records,
    choose_device,
    chunks,
    collect_image_items,
    compute_teacher_forced_metrics,
    core_ids_from_input_ids,
    decode_d4_label,
    edit_distance,
    greedy_decode_from_embeddings,
    ids_to_token_labels,
    load_model_checkpoint,
    make_image_pair_batch,
    plot_confidence_by_correctness,
    plot_confidence_by_domain,
    plot_domain_metrics,
    plot_exact_match_counts,
    plot_histogram,
    plot_metric_by_integer_field,
    plot_token_confusion,
    resolve_output_dir,
    resolve_path,
    rows_by_integer_field,
    safe_div,
    shifted_targets,
    trim_generated_ids,
    write_csv,
    write_json,
    write_jsonl,
    write_summary_md,
)
from src.model import ImageTransformPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sequence-completion behavior of a D4 autoregressive decoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--data_path", type=str, default="data", help="Dataset root with domain subfolders.")
    parser.add_argument("--split", type=str, default="heldout", help="train, val, validation, heldout, or test.")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_completion")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Image-pair batch size.")
    parser.add_argument("--val_size", type=float, default=None, help="Override config training.val_size.")
    parser.add_argument("--random_seed", type=int, default=None, help="Override config training.random_seed.")
    parser.add_argument("--max_images_per_domain", type=int, default=None, help="Limit held-out base images per domain.")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit final expanded image-pair examples.")
    parser.add_argument("--max_gen_len", type=int, default=None, help="Max generated continuation tokens.")
    parser.add_argument("--allowed_elements", nargs="*", default=None, help="D4 elements to evaluate. Default: all 8.")
    parser.add_argument("--target_mode", type=str, default=None, choices=["applied", "completion", "inverse"])
    parser.add_argument(
        "--negative_pairs_per_image",
        type=int,
        default=None,
        help="Deterministic unrelated/null pairs per held-out image. Default follows config data.negative_probability.",
    )
    parser.add_argument(
        "--prefix_strategy",
        type=str,
        default="all",
        choices=["all", "fixed", "random"],
        help="Which prefixes of the target sequence are given to the model.",
    )
    parser.add_argument("--prefix_len", type=int, default=0, help="Used when --prefix_strategy fixed.")
    parser.add_argument("--max_prefix_len", type=int, default=4, help="Max target-token prefix length for all/random.")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--save_jsonl", action="store_true", help="Also save predictions.jsonl.")
    return parser.parse_args()


def default_negative_pairs(config: Any, explicit_value: Optional[int]) -> int:
    if explicit_value is not None:
        return explicit_value
    return 1 if float(config.data.get("negative_probability", 0.0)) > 0.0 else 0


def select_prefix_lengths(
    core_len: int,
    strategy: str,
    fixed_prefix_len: int,
    max_prefix_len: int,
    rng: random.Random,
) -> List[int]:
    max_allowed = min(core_len, max_prefix_len)
    if strategy == "fixed":
        return [min(max(fixed_prefix_len, 0), core_len)]
    if strategy == "random":
        return [rng.randint(0, max_allowed)]
    return list(range(max_allowed + 1))


def content_ids_from_continuation(ids: List[int]) -> List[int]:
    content = []
    for token_id in ids:
        if token_id in {PAD_TOKEN_ID, START_TOKEN_ID}:
            continue
        if token_id == END_TOKEN_ID:
            break
        content.append(token_id)
    return content


def continuation_comparison(
    tokenizer: TransformTokenizer,
    predicted_continuation_ids: List[int],
    expected_continuation_ids: List[int],
) -> Dict[str, Any]:
    pred_trimmed = trim_generated_ids(predicted_continuation_ids)
    target_trimmed = list(expected_continuation_ids)
    max_len = max(len(pred_trimmed), len(target_trimmed))
    token_correct = 0
    for index in range(max_len):
        pred_id = pred_trimmed[index] if index < len(pred_trimmed) else None
        target_id = target_trimmed[index] if index < len(target_trimmed) else None
        token_correct += int(pred_id == target_id)

    pred_content_ids = content_ids_from_continuation(pred_trimmed)
    target_content_ids = content_ids_from_continuation(target_trimmed)
    pred_tokens = tokenizer.decode(pred_content_ids, skip_special_tokens=True)
    target_tokens = tokenizer.decode(target_content_ids, skip_special_tokens=True)
    distance = edit_distance(pred_tokens, target_tokens)

    return {
        "completion_exact_match": pred_trimmed == target_trimmed,
        "content_exact_match": pred_tokens == target_tokens,
        "next_token_correct": bool(pred_trimmed) and bool(target_trimmed) and pred_trimmed[0] == target_trimmed[0],
        "token_correct": token_correct,
        "token_total": max_len,
        "token_accuracy": safe_div(token_correct, max_len),
        "edit_distance": distance,
        "normalized_edit_distance": safe_div(distance, max(len(pred_tokens), len(target_tokens), 1)),
        "pred_continuation_tokens": pred_tokens,
        "target_continuation_tokens": target_tokens,
        "pred_token_labels": ids_to_token_labels(pred_trimmed),
        "target_token_labels": ids_to_token_labels(target_trimmed),
        "predicted_continuation_length": len(pred_tokens),
        "target_continuation_length": len(target_tokens),
    }


def build_completion_rows_for_batch(
    tokenizer: TransformTokenizer,
    model: ImageTransformPredictor,
    images_embeddings: torch.Tensor,
    batch,
    teacher_metrics: Dict[str, Any],
    max_seq_len: int,
    max_gen_len: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target_cpu = batch.target_ids.detach().cpu().tolist()
    core_ids_per_sample = [core_ids_from_input_ids(ids) for ids in target_cpu]

    prefix_jobs: Dict[int, List[int]] = {}
    for sample_index, core_ids in enumerate(core_ids_per_sample):
        for prefix_len in select_prefix_lengths(
            core_len=len(core_ids),
            strategy=args.prefix_strategy,
            fixed_prefix_len=args.prefix_len,
            max_prefix_len=args.max_prefix_len,
            rng=rng,
        ):
            prefix_jobs.setdefault(prefix_len, []).append(sample_index)

    for prefix_len, sample_indices in sorted(prefix_jobs.items()):
        prefix_rows = []
        expected_continuations: List[List[int]] = []
        for sample_index in sample_indices:
            core_ids = core_ids_per_sample[sample_index]
            prefix_rows.append([START_TOKEN_ID] + core_ids[:prefix_len])
            expected_continuations.append(core_ids[prefix_len:] + [END_TOKEN_ID])

        prefix_ids = torch.tensor(prefix_rows, dtype=torch.long, device=images_embeddings.device)
        subset_embeddings = images_embeddings[sample_indices]
        available_new_tokens = max_seq_len - prefix_ids.shape[1]
        if available_new_tokens <= 0:
            continue
        generated = greedy_decode_from_embeddings(
            model=model,
            images_embeddings=subset_embeddings,
            max_new_tokens=min(max_gen_len, available_new_tokens),
            prefix_ids=prefix_ids,
        )
        generated_cpu = generated.ids.detach().cpu().tolist()

        for local_index, sample_index in enumerate(sample_indices):
            record = batch.records[sample_index]
            generated_full_ids = generated_cpu[local_index]
            predicted_continuation_ids = generated_full_ids[prefix_ids.shape[1]:]
            comparison = continuation_comparison(
                tokenizer=tokenizer,
                predicted_continuation_ids=predicted_continuation_ids,
                expected_continuation_ids=expected_continuations[local_index],
            )

            prefix_core_ids = core_ids_per_sample[sample_index][:prefix_len]
            predicted_full_core_ids = prefix_core_ids + content_ids_from_continuation(
                trim_generated_ids(predicted_continuation_ids)
            )
            predicted_full_tokens = tokenizer.decode(predicted_full_core_ids, skip_special_tokens=True)
            decoded = decode_d4_label(predicted_full_tokens)

            full_sequence_exact = (
                bool(comparison["completion_exact_match"])
                and predicted_full_tokens == record.target_tokens
            )
            rows.append(
                {
                    "sample_id": f"{record.sample_id}:prefix={prefix_len}",
                    "base_sample_id": record.sample_id,
                    "domain": record.domain,
                    "image1": record.image1_path,
                    "image2": record.image2_path or f"<generated:{record.applied_element}>",
                    "is_negative": record.is_negative,
                    "applied_element": record.applied_element,
                    "applied_tokens": record.applied_tokens,
                    "target_label": record.target_element,
                    "target_tokens": record.target_tokens,
                    "input_prefix": tokenizer.decode(prefix_core_ids, skip_special_tokens=True),
                    "prefix_len": prefix_len,
                    "target_continuation": comparison["target_continuation_tokens"],
                    "predicted_continuation": comparison["pred_continuation_tokens"],
                    "predicted_full_tokens": predicted_full_tokens,
                    "predicted_label": str(decoded["label"]),
                    "predicted_tokens_canonical": decoded.get("canonical_tokens"),
                    "completion_exact_match": bool(comparison["completion_exact_match"]),
                    "full_sequence_exact": full_sequence_exact,
                    "content_exact_match": bool(comparison["content_exact_match"]),
                    "next_token_correct": bool(comparison["next_token_correct"]),
                    "token_accuracy": float(comparison["token_accuracy"]),
                    "token_correct": int(comparison["token_correct"]),
                    "token_total": int(comparison["token_total"]),
                    "confidence": generated.mean_confidences[local_index],
                    "mean_logprob": generated.mean_logprobs[local_index],
                    "teacher_loss": float(teacher_metrics["loss"][sample_index]),
                    "teacher_target_confidence": teacher_metrics["target_confidence"][sample_index],
                    "edit_distance": int(comparison["edit_distance"]),
                    "normalized_edit_distance": float(comparison["normalized_edit_distance"]),
                    "target_length": int(comparison["target_continuation_length"]),
                    "prediction_length": int(comparison["predicted_continuation_length"]),
                    "target_continuation_length": int(comparison["target_continuation_length"]),
                    "predicted_continuation_length": int(comparison["predicted_continuation_length"]),
                    "full_target_length": len(record.target_tokens),
                    "full_prediction_length": len(predicted_full_tokens),
                    "target_token_labels": comparison["target_token_labels"],
                    "pred_token_labels": comparison["pred_token_labels"],
                    "generated_token_ids": generated_full_ids,
                    "target_token_ids": target_cpu[sample_index],
                    "is_canonical_prediction": bool(decoded.get("is_canonical", False)),
                }
            )

    return rows


def add_completion_alias_metrics(overall_metrics: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    full_exact = sum(int(bool(row["full_sequence_exact"])) for row in rows)
    overall_metrics["completion_exact_accuracy"] = overall_metrics["sequence_accuracy"]
    overall_metrics["full_sequence_exact_accuracy"] = safe_div(full_exact, len(rows))
    overall_metrics["avg_true_continuation_length"] = overall_metrics["avg_target_length"]
    overall_metrics["avg_predicted_continuation_length"] = overall_metrics["avg_prediction_length"]


def evaluate(args: argparse.Namespace) -> None:
    config_path = str(resolve_path(args.config))
    checkpoint_path = str(resolve_path(args.checkpoint))
    data_path = str(resolve_path(args.data_path))
    output_dir = resolve_output_dir(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)
    device = choose_device(args.device, config)
    tokenizer = TransformTokenizer()
    model = ImageTransformPredictor(config.model)
    model.to(device)
    model.eval()
    load_model_checkpoint(model, checkpoint_path, device, config_path)

    batch_size = args.batch_size or int(config.training.get("batch_size", 8))
    val_size = args.val_size if args.val_size is not None else float(config.training.get("val_size", 0.1))
    random_seed = args.random_seed if args.random_seed is not None else int(config.training.get("random_seed", 42))
    rng = random.Random(random_seed)
    max_seq_len = int(config.model.decoder.max_seq_len)
    max_gen_len = args.max_gen_len or (max_seq_len - 1)
    target_mode = args.target_mode or config.data.get("target_mode", "completion")
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

    print("Completion evaluation")
    print(f"  encoder: {config.model.encoder.get('type', 'unknown')}")
    print(f"  target_mode: {target_mode}")
    print(f"  prefix_strategy: {args.prefix_strategy}")
    print(f"  held-out images: {len(image_items)}")
    print(f"  base eval pairs: {len(records)}")
    print(f"  output_dir: {output_dir}")

    prediction_rows: List[Dict[str, Any]] = []
    preprocess = model.image_pair_encoder.preprocess
    with torch.no_grad():
        for record_batch in tqdm(list(chunks(records, batch_size)), desc="Evaluating completion"):
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
            prediction_rows.extend(
                build_completion_rows_for_batch(
                    tokenizer=tokenizer,
                    model=model,
                    images_embeddings=images_embeddings,
                    batch=batch,
                    teacher_metrics=teacher_metrics,
                    max_seq_len=max_seq_len,
                    max_gen_len=max_gen_len,
                    args=args,
                    rng=rng,
                )
            )

    if not prediction_rows:
        raise RuntimeError("No completion rows were produced. Check prefix settings and max_seq_len.")

    overall_metrics, domain_rows, class_rows, token_rows, _, token_confusion = aggregate_rows(
        prediction_rows,
        exact_key="completion_exact_match",
        class_labels=ELEMENT_CLASS_LABELS,
    )
    add_completion_alias_metrics(overall_metrics, prediction_rows)
    metrics_by_prefix_len = rows_by_integer_field(
        prediction_rows,
        "prefix_len",
        "completion_exact_match",
    )
    metrics_by_target_cont_len = rows_by_integer_field(
        prediction_rows,
        "target_continuation_length",
        "completion_exact_match",
    )

    run_config = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "data_path": data_path,
        "split": args.split,
        "target_mode": target_mode,
        "encoder": config.model.encoder.get("type", "unknown"),
        "batch_size": batch_size,
        "max_gen_len": max_gen_len,
        "negative_pairs_per_image": negative_pairs,
        "prefix_strategy": args.prefix_strategy,
        "prefix_len": args.prefix_len,
        "max_prefix_len": args.max_prefix_len,
        "heldout_images": len(image_items),
        "base_eval_pairs": len(records),
        "completion_examples": len(prediction_rows),
    }
    metric_notes = [
        "Completion Exact Accuracy checks whether the generated continuation exactly matches the true remaining tokens plus EOS.",
        "Token Accuracy is position-wise over continuation tokens; EOS and [NULL] are included, PAD/BOS are excluded.",
        "Precision/Recall/F1 are token-level and D4-label-level macro/micro/weighted reports. Token reports include EOS and [NULL].",
        "Average confidence is the mean probability of greedily selected generated tokens. Separate exact/error confidence is saved in metrics.json.",
        "Current project has image-pair conditioning, so the decoder completes a prefix while conditioned on held-out image-pair embeddings.",
    ]
    metrics_payload = {
        "run": run_config,
        "metric_notes": metric_notes,
        "overall": overall_metrics,
        "by_domain": domain_rows,
        "by_prefix_len": metrics_by_prefix_len,
        "by_target_continuation_length": metrics_by_target_cont_len,
    }

    write_json(str(Path(output_dir) / "metrics.json"), metrics_payload)
    write_csv(str(Path(output_dir) / "metrics_by_domain.csv"), domain_rows)
    write_csv(str(Path(output_dir) / "metrics_by_prefix_len.csv"), metrics_by_prefix_len)
    write_csv(str(Path(output_dir) / "metrics_by_target_continuation_len.csv"), metrics_by_target_cont_len)
    write_csv(str(Path(output_dir) / "metrics_by_class.csv"), class_rows)
    write_csv(str(Path(output_dir) / "metrics_by_token.csv"), token_rows)
    write_csv(str(Path(output_dir) / "predictions.csv"), prediction_rows)
    if args.save_jsonl:
        write_jsonl(str(Path(output_dir) / "predictions.jsonl"), prediction_rows)
    write_summary_md(
        path=str(Path(output_dir) / "summary.md"),
        title="Completion Model Evaluation",
        config_payload=run_config,
        overall_metrics=overall_metrics,
        metric_notes=metric_notes,
    )

    if not args.no_plots:
        plot_domain_metrics(
            output_dir=output_dir,
            domain_rows=domain_rows,
            metrics=["sequence_accuracy", "token_accuracy", "class_f1_macro", "avg_confidence"],
            title="Качество completion-модели по доменам",
        )
        plot_confidence_by_domain(output_dir, domain_rows)
        plot_exact_match_counts(output_dir, prediction_rows, exact_key="completion_exact_match")
        plot_confidence_by_correctness(output_dir, prediction_rows, exact_key="completion_exact_match")
        plot_histogram(
            output_dir,
            [float(row["confidence"]) for row in prediction_rows if row.get("confidence") is not None],
            output_name="confidence_distribution",
            title="Распределение confidence",
            xlabel="Mean chosen-token probability",
        )
        plot_histogram(
            output_dir,
            [float(row["target_continuation_length"]) for row in prediction_rows],
            output_name="target_continuation_length_distribution",
            title="Распределение длины истинного continuation",
            xlabel="True continuation length",
            bins=max(5, min(20, max(int(row["target_continuation_length"]) for row in prediction_rows) + 1)),
        )
        plot_histogram(
            output_dir,
            [float(row["predicted_continuation_length"]) for row in prediction_rows],
            output_name="predicted_continuation_length_distribution",
            title="Распределение длины предсказанного continuation",
            xlabel="Predicted continuation length",
            bins=max(5, min(20, max(int(row["predicted_continuation_length"]) for row in prediction_rows) + 1)),
        )
        plot_metric_by_integer_field(
            output_dir,
            prediction_rows,
            field_name="target_continuation_length",
            exact_key="completion_exact_match",
            output_name="quality_by_true_continuation_length",
            title="Качество от длины истинного continuation",
        )
        plot_metric_by_integer_field(
            output_dir,
            prediction_rows,
            field_name="prefix_len",
            exact_key="completion_exact_match",
            output_name="quality_by_prefix_len",
            title="Качество от длины prefix",
        )
        plot_token_confusion(output_dir, token_confusion)

    print("\nSaved:")
    print(f"  {Path(output_dir) / 'metrics.json'}")
    print(f"  {Path(output_dir) / 'metrics_by_domain.csv'}")
    print(f"  {Path(output_dir) / 'metrics_by_prefix_len.csv'}")
    print(f"  {Path(output_dir) / 'predictions.csv'}")
    print(f"  {Path(output_dir) / 'summary.md'}")
    if not args.no_plots:
        print(f"  {Path(output_dir) / 'plots'}")
    print("\nMain metrics:")
    print(f"  completion_exact_accuracy: {overall_metrics['completion_exact_accuracy']:.4f}")
    print(f"  token_accuracy: {overall_metrics['token_accuracy']:.4f}")
    print(f"  next_token_accuracy: {overall_metrics['next_token_accuracy']:.4f}")
    print(f"  token_f1_macro: {overall_metrics['token_f1_macro']:.4f}")
    print(f"  avg_confidence: {overall_metrics['avg_confidence']:.4f}")
    print(f"  avg_true_continuation_length: {overall_metrics['avg_true_continuation_length']:.4f}")
    print(f"  avg_predicted_continuation_length: {overall_metrics['avg_predicted_continuation_length']:.4f}")


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
