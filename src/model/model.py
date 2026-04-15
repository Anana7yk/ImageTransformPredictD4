import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Tuple, List

from .decoder import TransformDecoder
from .efficientnet_encoder import ImagePairEncoderEfficientNet
from .vit_encoder import ImagePairEncoderViT


class ImageTransformPredictor(nn.Module):
    """
    Complete end-to-end model for predicting image transformation sequences from image pairs.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        encoder_type = config.encoder.get("type", "efficientnet_encoder")
        if encoder_type in ("efficientnet", "efficientnet_encoder"):
            self.image_pair_encoder = ImagePairEncoderEfficientNet(config.encoder)
        elif encoder_type in ("vit", "vit_encoder"):
            self.image_pair_encoder = ImagePairEncoderViT(config.encoder)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        self.bos_token_id = config.decoder.bos_token_id
        self.eos_token_id = config.decoder.eos_token_id
        self.pad_token_id = config.decoder.pad_token_id
        
        self.transform_decoder = TransformDecoder(config.decoder)

    def extract_image_embeddings(self, image_batch_1, image_batch_2):
        """
        Extract embeddings for two batches of images.
        Args:
            image_batch_1 (torch.Tensor): First batch of images with shape [batch_size, 3, 224, 224].
            image_batch_2 (torch.Tensor): Second batch of images with shape [batch_size, 3, 224, 224].
        Returns:
            tuple: (features_1, features_2) two batches of embeddings.
        """
        return self.image_pair_encoder.extract_image_embeddings(image_batch_1, image_batch_2)

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        idx: torch.LongTensor,
        use_precomputed_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        """
        images_embeddings = self.image_pair_encoder(image_batch_1, image_batch_2, use_precomputed_embeddings)

        targets = torch.roll(idx, shifts=-1, dims=1)
        targets[:, -1] = self.pad_token_id
        logits, loss = self.transform_decoder(
            idx=idx,
            images_embeddings=images_embeddings,
            targets=targets
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ) -> torch.LongTensor:
        """Autoregressive generation."""
        if max_new_tokens is None:
            max_new_tokens = self.config.decoder.max_seq_len - 1

        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if bos_token_id is None:
            bos_token_id = self.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        images_embeddings = self.image_pair_encoder(image_batch_1, image_batch_2)
        return self.transform_decoder.generate(
            images_embeddings=images_embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )

    @torch.no_grad()
    def generate_step_with_cross_attn(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        idx_prefix: Optional[torch.Tensor] = None,
        use_precomputed_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Run one autoregressive generation step and return cross-attention weights.
        
        Args:
            image_batch_1, image_batch_2: [B, 3, H, W] or precomputed embeddings if use_precomputed_embeddings=True
            idx_prefix: [B, T] token IDs (e.g., [BOS]). If None → starts with [BOS].
            use_precomputed_embeddings: if True, treats img batches as [B, L, D] embeddings.

        Returns:
            next_logits: [B, vocab_size]
            next_token: [B]
            cross_attn_per_layer: list of length n_layer, each: [B, n_head, L_key]
                where L_key = 395 for ViT-B/16 (197 + 1 + 197)
        """
        if idx_prefix is None:
            device = image_batch_1.device if not use_precomputed_embeddings else image_batch_1.device
            B = image_batch_1.shape[0]
            idx_prefix = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

        images_embeddings = self.image_pair_encoder(
            image_batch_1, image_batch_2, use_precomputed_embeddings=use_precomputed_embeddings
        )

        return self.transform_decoder.generate_step_with_cross_attn(
            images_embeddings=images_embeddings,
            idx_prefix=idx_prefix,
        )
