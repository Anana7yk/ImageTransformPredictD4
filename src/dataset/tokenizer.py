from typing import List, Union, Optional
import torch

# --- Vocabulary definition ---
D4_CANONICAL_SEQUENCES = {
    "e": ["e"],
    "r": ["r"],
    "r2": ["r", "r"],
    "r3": ["r", "r", "r"],
    "s": ["s"],
    "sr": ["s", "r"],
    "sr2": ["s", "r", "r"],
    "sr3": ["s", "r", "r", "r"],
}

NULL_SEQUENCE = ["[NULL]"]

TRANSFORM_TOKENS = {
    "[PAD]": 0,
    "[START]": 1,
    "[END]": 2,
    "e": 3,
    "r": 4,
    "s": 5,
    "[NULL]": 6,
}

ID_TO_TOKEN = {v: k for k, v in TRANSFORM_TOKENS.items()}
VOCAB_SIZE = len(TRANSFORM_TOKENS)

PAD_TOKEN_ID = TRANSFORM_TOKENS["[PAD]"]
START_TOKEN_ID = TRANSFORM_TOKENS["[START]"]
END_TOKEN_ID = TRANSFORM_TOKENS["[END]"]
NULL_TOKEN_ID = TRANSFORM_TOKENS["[NULL]"]


def get_canonical_sequence(transform_name: str) -> List[str]:
    if transform_name == "[NULL]":
        return list(NULL_SEQUENCE)
    if transform_name not in D4_CANONICAL_SEQUENCES:
        raise ValueError(f"Unknown D4 transform: {transform_name}")
    return list(D4_CANONICAL_SEQUENCES[transform_name])


class TransformTokenizer:
    """
    Minimal tokenizer for canonical D4 transformation sequences.
    
    Supports:
      - Encoding list of transform names to token IDs
      - Optional padding to `max_seq_len`
      - Optional generation of targets (shifted input for autoregressive training)
    """

    def __init__(self):
        self.vocab = TRANSFORM_TOKENS
        self.ids_to_tokens = ID_TO_TOKEN

    def encode(
        self,
        transforms: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_seq_len: Optional[int] = None,
        return_targets: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a sequence of transform names into token IDs.

        Args:
            transforms: List of transform names (e.g., ["s", "r", "r"]) or space-separated string.
            add_special_tokens: Whether to add [START] and [END] tokens.
            max_seq_len: If provided, pad or truncate to this length.
            return_targets: If True, also return targets = input shifted left by 1 (for training).

        Returns:
            If return_targets=False: `input_ids` as [T] or [max_seq_len] torch.LongTensor.
            If return_targets=True: tuple (`input_ids`, `targets`), both tensors of same shape.
        """
        if isinstance(transforms, str):
            transforms = transforms.split()

        # Convert to IDs
        token_ids = [self.vocab.get(t, PAD_TOKEN_ID) for t in transforms]

        # Add special tokens
        if add_special_tokens:
            token_ids = [START_TOKEN_ID] + token_ids + [END_TOKEN_ID]

        # Truncate if needed
        if max_seq_len is not None:
            token_ids = token_ids[:max_seq_len]

        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # Pad if needed
        if max_seq_len is not None and len(token_ids) < max_seq_len:
            padding = torch.full(
                (max_seq_len - len(token_ids),),
                PAD_TOKEN_ID,
                dtype=torch.long
            )
            input_ids = torch.cat([input_ids, padding], dim=0)

        if not return_targets:
            return input_ids

        # Create targets: shift left by 1
        targets = torch.full_like(input_ids, PAD_TOKEN_ID)
        if input_ids.shape[0] > 1:
            targets[:-1] = input_ids[1:]

        return input_ids, targets

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode token IDs back to transform names.

        Args:
            token_ids: 1D list or tensor of token IDs.
            skip_special_tokens: Whether to skip [PAD], [START], [END].

        Returns:
            List of transform names (strings).
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        special_ids = {PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID}
        tokens = []
        for tid in token_ids:
            token = self.ids_to_tokens.get(tid, "[UNK]")
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(token)
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        import json
        vocab_file = f"{pretrained_model_name_or_path}/vocab.json"
        with open(vocab_file, 'r') as f:
            global TRANSFORM_TOKENS
            TRANSFORM_TOKENS = json.load(f)
        global ID_TO_TOKEN
        ID_TO_TOKEN = {v: k for k, v in TRANSFORM_TOKENS.items()}
        return cls()
