import os
import random
from typing import List, Optional, Tuple, Callable, Dict
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .tokenizer import TransformTokenizer, NULL_SEQUENCE
from .augmentation import ImageTransformer


def _default_image_preprocessor() -> Callable[[Image.Image], torch.Tensor]:
    """Default image preprocessor: resize to 224x224 + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DomainNetDataset(Dataset):
    """
    DomainNet dataset with per-domain train/val split.

    Returns:
        - image_1 (tensor)
        - image_2 (tensor)
        - tokenized canonical D4 target sequence or null token (tensor)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: TransformTokenizer,
        transformer: ImageTransformer,
        split: str = "train",
        val_size: float = 0.1,
        image_preprocessor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        random_seed: int = 42,
        max_seq_len: int = 6,
        negative_probability: float = 0.5,
        return_seq: bool = True
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'!")

        self.data_dir = data_dir
        self.preprocessor = image_preprocessor or _default_image_preprocessor()
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.split = split
        self.max_seq_len = max_seq_len
        self.negative_probability = negative_probability
        self.return_seq = return_seq

        domain_to_paths: Dict[str, List[str]] = {}
        for domain in sorted(os.listdir(data_dir)):
            domain_path = os.path.join(data_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            paths = []
            for file in os.listdir(domain_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(domain_path, file))
            if paths:
                domain_to_paths[domain] = paths

        self.image_paths = []
        rng = random.Random(random_seed)

        for domain, paths in domain_to_paths.items():
            shuffled = rng.sample(paths, len(paths))
            n_val = int(len(shuffled) * val_size)

            if split == "val":
                self.image_paths.extend(shuffled[:n_val])
            else:  # train
                self.image_paths.extend(shuffled[n_val:])

        if split == "train":
            self.image_paths = rng.sample(self.image_paths, len(self.image_paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def _sample_negative_image(self, current_idx: int) -> Image.Image:
        if len(self.image_paths) < 2:
            raise ValueError("At least two images are required to sample a negative pair.")

        negative_idx = current_idx
        while negative_idx == current_idx:
            negative_idx = random.randrange(len(self.image_paths))

        negative_image_path = self.image_paths[negative_idx]
        return Image.open(negative_image_path).convert("RGB")

    def _sample_pair(
        self,
        original_image: Image.Image,
        idx: int,
    ) -> Tuple[Image.Image, List[str]]:
        use_negative_pair = self.return_seq and len(self.image_paths) > 1 and random.random() < self.negative_probability

        if use_negative_pair:
            return self._sample_negative_image(idx), list(NULL_SEQUENCE)

        transformed_image, transform_sequence = self.transformer.transform(original_image)
        return transformed_image, transform_sequence

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        original_image = Image.open(image_path).convert("RGB")

        transformed_image, transform_sequence = self._sample_pair(original_image, idx)

        original_tensor = self.preprocessor(original_image)
        transformed_tensor = self.preprocessor(transformed_image.convert("RGB"))
        if self.return_seq:
            sequence_ids = self.tokenizer.encode(
                transforms=transform_sequence,
                add_special_tokens=True,
                max_seq_len=self.max_seq_len,
                return_targets=False,
            )
            
            return original_tensor, transformed_tensor, sequence_ids
        else:
            return original_tensor, transformed_tensor


def get_domainnet_dataloaders(
    data_dir: str,
    tokenizer: TransformTokenizer,
    transformer: ImageTransformer,
    batch_size: int = 32,
    num_workers: int = 4,
    val_size: float = 0.1,
    image_preprocessor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    random_seed: int = 42,
    max_seq_len: int = 6,
    negative_probability: float = 0.5,
    return_seq: bool = True
):
    common_kwargs = dict(
        data_dir=data_dir,
        tokenizer=tokenizer,
        transformer=transformer,
        val_size=val_size,
        image_preprocessor=image_preprocessor,
        random_seed=random_seed,
        max_seq_len=max_seq_len,
        negative_probability=negative_probability,
        return_seq=return_seq
    )

    train_dataset = DomainNetDataset(split="train", **common_kwargs)
    val_dataset = DomainNetDataset(split="val", **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader}


class SimpleDomainNetDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_size: float = 0.1,
        random_seed: int = 42,
    ):
        if split not in ("train", "val"):
            raise ValueError("Split must be 'train' or 'val'!")

        self.data_dir = data_dir
        self.split = split
        self.domain_distribution: Dict[str, int] = {}
        self.image_paths: List[Tuple[str, str]] = []
        self._collect_images_and_domains(val_size, random_seed)
        self._calculate_domain_distribution()

    def _collect_images_and_domains(self, val_size: float, random_seed: int) -> None:
        domain_to_paths: Dict[str, List[Tuple[str, str]]] = {}
        rng = random.Random(random_seed)

        for domain in sorted(os.listdir(self.data_dir)):
            domain_path = os.path.join(self.data_dir, domain)
            if not os.path.isdir(domain_path):
                continue

            paths_with_domains = []
            for file in os.listdir(domain_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths_with_domains.append((os.path.join(domain_path, file), domain))

            if paths_with_domains:
                domain_to_paths[domain] = paths_with_domains

        for domain, paths in domain_to_paths.items():
            shuffled = rng.sample(paths, len(paths))
            n_val = int(len(shuffled) * val_size)

            if self.split == "val":
                self.image_paths.extend(shuffled[:n_val])
            else:
                self.image_paths.extend(shuffled[n_val:])

        if self.split == "train":
            self.image_paths = rng.sample(self.image_paths, len(self.image_paths))

    def _calculate_domain_distribution(self) -> None:
        for _, domain in self.image_paths:
            self.domain_distribution[domain] = self.domain_distribution.get(domain, 0) + 1

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        image_path, domain = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return image, domain
