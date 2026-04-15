import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from omegaconf import DictConfig


class EfficientNetEncoder(nn.Module):
    """
    Fixed-dimension image encoder based on pretrained EfficientNet-B3.

    This module extracts global image features using EfficientNet-B3 with the classifier head removed.
    It always outputs 1536-dimensional embeddings reshaped as a sequence of length 1:
    [batch_size, 1, 1536], to match the interface of ViT-style encoders.

    Includes a built-in preprocessing pipeline compatible with EfficientNet.

    Attributes:
        preprocess (transforms.Compose): Standard preprocessing for EfficientNet inputs.
        feature_dim (int): Fixed output dimension (1536).
    """

    def __init__(self, freeze=True):
        super().__init__()

        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        self.feature_dim = 1536

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, 1536]
        return features.unsqueeze(1)  # [batch_size, 1, 1536]


class ImagePairEncoderEfficientNet(nn.Module):
    """
    Encoder for paired images that produces a fused embedding as a sequence of length 1.

    Processes two images, extracts 1536-dim features from each, concatenates them (3072-dim),
    and projects into a token embedding space of dimension `out_token_n_embd`.

    Output shape: [batch_size, 1, out_token_n_embd] — compatible with transformer decoders
    that expect `n_embd`-dimensional token embeddings.

    Attributes:
        image_encoder (EfficientNetEncoder): Shared image feature extractor.
        fuser (nn.Linear): Projects concatenated features to `out_token_n_embd`.
        output_dim (int): Equals `out_token_n_embd`.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.image_encoder = EfficientNetEncoder(freeze=config.freeze_image_encoder)

        if not hasattr(config, 'out_token_n_embd'):
            raise ValueError("Config must contain 'out_token_n_embd' (embedding dimension for output token)")

        concat_dim = self.image_encoder.feature_dim * 2  # 1536 * 2 = 3072
        self.fuser = nn.Linear(concat_dim, config.out_token_n_embd)
        self.output_dim = config.out_token_n_embd

    @property
    def preprocess(self):
        return self.image_encoder.preprocess

    def extract_image_embeddings(self, image_batch_1, image_batch_2):
        features_1 = self.image_encoder(image_batch_1)  # [B, 1, 1536]
        features_2 = self.image_encoder(image_batch_2)  # [B, 1, 1536]
        return features_1, features_2

    def forward(self, image_batch_1, image_batch_2, use_precomputed_embeddings=False):
        if not use_precomputed_embeddings:
            features_1, features_2 = self.extract_image_embeddings(image_batch_1, image_batch_2)
        else:
            features_1, features_2 = image_batch_1, image_batch_2  # assumed [B, 1, 1536]

        concatenated = torch.cat([features_1.squeeze(1), features_2.squeeze(1)], dim=1)  # [B, 3072]
        fused = self.fuser(concatenated)                           # [B, out_token_n_embd]
        return fused.unsqueeze(1)  # [B, 1, out_token_n_embd]
