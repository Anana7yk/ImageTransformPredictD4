#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from src.dataset import END_TOKEN_ID, PAD_TOKEN_ID, START_TOKEN_ID, TransformTokenizer
from src.model import ImageTransformPredictor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config_d4.yaml"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def choose_device(device_arg: Optional[str], config) -> torch.device:
    requested = device_arg or config.training.get("device", "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA device is not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_pil_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def build_image_tensor(image: Image.Image, size: int = 224) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def normalize_image(unit_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=device, dtype=unit_tensor.dtype)
    std = IMAGENET_STD.to(device=device, dtype=unit_tensor.dtype)
    return (unit_tensor - mean) / std


def denormalize_image(unit_tensor: torch.Tensor) -> torch.Tensor:
    return unit_tensor.clamp(0.0, 1.0)


def tensor_to_pil(unit_tensor: torch.Tensor) -> Image.Image:
    image = unit_tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    return transforms.ToPILImage()(image)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    dh = image[:, :, 1:, :] - image[:, :, :-1, :]
    dw = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dh.abs().mean() + dw.abs().mean()


def normalize_target_spec(target_spec: str) -> List[str]:
    raw = target_spec.strip()
    aliases = {
        "r^2": "r r",
        "r^3": "r r r",
        "sr^2": "s r r",
        "sr^3": "s r r r",
    }
    raw = aliases.get(raw.replace(" ", ""), raw)
    tokens = raw.replace(",", " ").split()
    if not tokens:
        raise ValueError("Target must not be empty.")
    for token in tokens:
        if token not in {"e", "r", "s", "[NULL]"}:
            raise ValueError(f"Unsupported target token: {token}")
    return tokens


def build_input_and_targets(
    tokenizer: TransformTokenizer,
    tokens: List[str],
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = tokenizer.encode(
        transforms=tokens,
        add_special_tokens=True,
        max_seq_len=max_seq_len,
        return_targets=False,
    ).unsqueeze(0).to(device)
    targets = torch.full_like(idx, PAD_TOKEN_ID)
    targets[:, :-1] = idx[:, 1:]
    return idx, targets


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
def greedy_decode(
    model: ImageTransformPredictor,
    image1_norm: torch.Tensor,
    image2_norm: torch.Tensor,
    tokenizer: TransformTokenizer,
    max_gen_len: int,
) -> List[str]:
    images_embeddings = model.image_pair_encoder(image1_norm, image2_norm)
    idx = torch.full((1, 1), model.bos_token_id, dtype=torch.long, device=image1_norm.device)

    for _ in range(max_gen_len):
        logits, _ = model.transform_decoder(idx=idx, images_embeddings=images_embeddings, targets=None)
        next_logits = logits[:, -1, :].clone()
        next_logits[:, PAD_TOKEN_ID] = -float("inf")
        next_logits[:, START_TOKEN_ID] = -float("inf")
        next_id = int(torch.argmax(next_logits, dim=-1).item())
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=image1_norm.device)], dim=1)
        if next_id == END_TOKEN_ID:
            break

    return tokenizer.decode(idx[0], skip_special_tokens=False)


def default_output_path(image_path: str, target_spec: str) -> str:
    image = Path(image_path)
    suffix = target_spec.replace(" ", "_").replace("^", "")
    return str(image.with_name(f"{image.stem}__optimized_{suffix}.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a second image tensor through TransformDecoder.forward so that the current "
            "checkpoint assigns high likelihood to a requested target token sequence."
        )
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image1", type=str, required=True)
    parser.add_argument("--target", type=str, required=True, help='Examples: "r", "r r r", "r^3", "s r".')
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tv_weight", type=float, default=0.001)
    parser.add_argument("--l2_weight", type=float, default=0.01)
    parser.add_argument("--init", type=str, choices=("source", "noise"), default="source")
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    image1_path = _resolve_project_path(args.image1)
    output_path = _resolve_project_path(args.output) if args.output else default_output_path(args.image1, args.target)

    config = OmegaConf.load(config_path)
    device = choose_device(args.device, config)

    model = ImageTransformPredictor(config.model)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_model_checkpoint(model, checkpoint_path, device, config_path)

    tokenizer = TransformTokenizer()
    target_tokens = normalize_target_spec(args.target)
    idx, targets = build_input_and_targets(
        tokenizer=tokenizer,
        tokens=target_tokens,
        max_seq_len=config.model.decoder.max_seq_len,
        device=device,
    )

    base_image = load_pil_image(image1_path)
    image1_unit = build_image_tensor(base_image).to(device)
    image1_norm = normalize_image(image1_unit, device)

    if args.init == "source":
        init_unit = image1_unit.clone()
    else:
        init_unit = torch.rand_like(image1_unit)

    init_unit = init_unit.clamp(1e-4, 1.0 - 1e-4)
    image2_logits = torch.logit(init_unit).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([image2_logits], lr=args.lr)

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"image1: {image1_path}")
    print(f"encoder: {config.model.encoder.type}")
    print(f"target tokens: {target_tokens}")
    print(f"steps: {args.steps}")
    print(f"lr: {args.lr}")
    print(f"tv_weight: {args.tv_weight}")
    print(f"l2_weight: {args.l2_weight}")
    print(f"init: {args.init}")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        image2_unit = torch.sigmoid(image2_logits)
        image2_norm = normalize_image(image2_unit, device)

        images_embeddings = model.image_pair_encoder(image1_norm, image2_norm)
        logits, ce_loss = model.transform_decoder(
            idx=idx,
            images_embeddings=images_embeddings,
            targets=targets,
        )

        tv_loss = total_variation_loss(image2_unit)
        l2_loss = F.mse_loss(image2_unit, image1_unit)
        loss = ce_loss + args.tv_weight * tv_loss + args.l2_weight * l2_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image2_logits.clamp_(-8.0, 8.0)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
                pred_tokens = tokenizer.decode(pred_ids, skip_special_tokens=False)
            print(
                f"step {step:04d} | total_loss={loss.item():.4f} | "
                f"ce={ce_loss.item():.4f} | tv={tv_loss.item():.4f} | "
                f"l2={l2_loss.item():.4f} | argmax_tokens={pred_tokens}"
            )

    with torch.no_grad():
        final_unit = denormalize_image(torch.sigmoid(image2_logits))
        final_norm = normalize_image(final_unit, device)
        decoded_tokens = greedy_decode(
            model=model,
            image1_norm=image1_norm,
            image2_norm=final_norm,
            tokenizer=tokenizer,
            max_gen_len=config.model.decoder.max_seq_len - 1,
        )

    tensor_to_pil(final_unit).save(output_path)
    print("")
    print(f"saved output image: {output_path}")
    print(f"greedy decoded tokens on optimized pair: {decoded_tokens}")


if __name__ == "__main__":
    main()
