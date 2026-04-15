#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from src.dataset import ImageTransformer
from src.dataset import TransformTokenizer
from src.dataset import get_domainnet_dataloaders
from src.model import ImageTransformPredictor
from src.train import train_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config_d4.yaml"


def _resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def main(data_path: str, config_path: str):
    train_config = OmegaConf.load(config_path)
    train_config.training.checkpoint_dir = _resolve_project_path(train_config.training.checkpoint_dir)
    train_config.data.tensorboard_logdir = _resolve_project_path(train_config.data.tensorboard_logdir)

    model = ImageTransformPredictor(train_config.model)
    tokenizer = TransformTokenizer()
    transformer = ImageTransformer(
        allowed_elements=train_config.data.get("allowed_d4_elements"),
        target_mode=train_config.data.get("target_mode", "completion"),
    )

    dataloaders = get_domainnet_dataloaders(
        data_path,
        tokenizer,
        transformer,
        batch_size=train_config.training.batch_size,
        num_workers=train_config.data.get('num_workers', 4),
        val_size=train_config.training.val_size,
        random_seed=train_config.training.random_seed,
        max_seq_len=train_config.model.decoder.max_seq_len,
        negative_probability=train_config.data.get('negative_probability', 0.5),
        image_preprocessor=model.image_pair_encoder.preprocess,
    )

    train_model(
        model,
        dataloaders['train'],
        dataloaders['val'],
        train_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Image Transform Predictor.')
    parser.add_argument(
        '--data_path',
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help='Path to the dataset directory with domain subfolders.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG),
        help='Path to the training config YAML.'
    )
    args = parser.parse_args()
    main(args.data_path, args.config)
