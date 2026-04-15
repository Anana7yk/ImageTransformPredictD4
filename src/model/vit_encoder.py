import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Compose
from omegaconf import DictConfig


class ViTImageEncoder(nn.Module):
    """
    Image encoder based on Vision Transformer (ViT) from torchvision.

    Extracts a full sequence of patch and class tokens from input images using a pretrained
    ViT-B/16 model. The output includes all tokens before the final classification head,
    preserving spatial and semantic information as a sequence.

    Attributes:
        preprocess (Compose): ImageNet-compatible preprocessing pipeline.
        feature_dim (int): Dimensionality of each token (e.g., 768 for ViT-B/16).
    """

    def __init__(self, freeze=True):
        """
        Initializes the ViT encoder.

        Args:
            freeze (bool): If True, freezes all parameters of the ViT backbone.
                           Defaults to True.
        """
        super().__init__()

        # Load pretrained ViT-B/16
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)

        # Remove classification head
        self.vit.heads = nn.Identity()

        # Freeze backbone if needed
        for param in self.vit.parameters():
            param.requires_grad = not freeze

        # Feature dimension = hidden size of ViT
        self.feature_dim = self.vit.hidden_dim  # 768 for ViT-B/16

        # Official preprocessing
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @property
    def preprocess(self) -> Compose:
        """Returns the ImageNet-compatible preprocessing pipeline."""
        return self.preprocess_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of images into a sequence of tokens.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224],
                              normalized with ImageNet statistics.

        Returns:
            torch.Tensor: Token embeddings of shape [batch_size, num_tokens, feature_dim],
                          where num_tokens = 196 (patches) + 1 (class token) = 197.
        """
        # Patch embedding
        n = x.shape[0]
        x = self.vit._process_input(x)  # [B, C, H, W] -> [B, N, D]

        # Expand class token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)  # [B, 1, D]
        x = torch.cat([batch_class_token, x], dim=1)  # [B, N+1, D]

        # Apply transformer
        x = self.vit.encoder(x)  # [B, N+1, D]

        return x  # All tokens, no projection

    def extract_image_embeddings(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Extracts token-level embeddings from a batch of images."""
        return self.forward(image_batch)


class ImagePairEncoderViT(nn.Module):
    """
    Encoder for pairs of images using a pretrained Vision Transformer (ViT).

    Each image is independently encoded and projected into the decoder's token embedding space
    of dimension `out_token_n_embd`. A learnable separator token (in the target space) is inserted
    between the two sequences to form a single combined token sequence.

    Output shape: [batch_size, L1 + 1 + L2, out_token_n_embd], where L1 = L2 = 197.

    Attributes:
        preprocess (Compose): ImageNet-compatible preprocessing pipeline.
        output_dim (int): Equals `out_token_n_embd` from the configuration.
    """

    def __init__(self, config: DictConfig):
        """
        Initializes the paired image encoder.

        Args:
            config (DictConfig): Configuration object containing:
                - freeze_image_encoder (bool): Whether to freeze the ViT backbone.
                - out_token_n_embd (int): Target dimension for output tokens.
        """
        super().__init__()
        self.config = config

        self.image_encoder = ViTImageEncoder(freeze=config.freeze_image_encoder)

        if not hasattr(config, 'out_token_n_embd'):
            raise ValueError("Config must contain 'out_token_n_embd'")

        # Projection to decoder's token space
        self.proj = nn.Linear(self.image_encoder.feature_dim, config.out_token_n_embd)

        # Learnable separator in target space
        self.sep_token = nn.Parameter(torch.randn(1, 1, config.out_token_n_embd))

        self.output_dim = config.out_token_n_embd

    @property
    def preprocess(self) -> Compose:
        """Returns the ImageNet-compatible preprocessing pipeline."""
        return self.image_encoder.preprocess

    def extract_image_embeddings(self, image_batch_1: torch.Tensor, image_batch_2: torch.Tensor):
        """
        Encodes two image batches and projects their tokens into the decoder's embedding space.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Each of shape [B, 197, out_token_n_embd].
        """
        tokens1 = self.image_encoder(image_batch_1)  # [B, 197, D]
        tokens2 = self.image_encoder(image_batch_2)  # [B, 197, D]

        proj1 = self.proj(tokens1)  # [B, 197, E]
        proj2 = self.proj(tokens2)  # [B, 197, E]
        return proj1, proj2

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        use_precomputed_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Encodes a pair of images into a combined token sequence.

        Args:
            image_batch_1, image_batch_2:
                If use_precomputed_embeddings=False: [B, 3, 224, 224] (ImageNet-normalized).
                If True: [B, 197, out_token_n_embd] (already projected tokens).
            use_precomputed_embeddings: Skip encoder and projection if True.

        Returns:
            torch.Tensor: [B, 197 + 1 + 197, out_token_n_embd] = [B, 395, out_token_n_embd]
        """
        if not use_precomputed_embeddings:
            proj1, proj2 = self.extract_image_embeddings(image_batch_1, image_batch_2)
        else:
            proj1, proj2 = image_batch_1, image_batch_2

        B = proj1.shape[0]
        sep = self.sep_token.expand(B, -1, -1)  # [B, 1, out_token_n_embd]
        combined = torch.cat([proj1, sep, proj2], dim=1)  # [B, 395, out_token_n_embd]

        return combined