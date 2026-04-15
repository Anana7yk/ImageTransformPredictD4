import random
from typing import List, Tuple, Optional
from PIL import Image, ImageOps

from .tokenizer import D4_CANONICAL_SEQUENCES


IDENTITY_MATRIX = ((1, 0), (0, 1))
ROTATE_MATRIX = ((0, -1), (1, 0))
REFLECT_MATRIX = ((-1, 0), (0, 1))


def _matmul(a, b):
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ),
    )


def _apply_generator_to_matrix(current, token: str):
    if token == "e":
        return current
    if token == "r":
        return _matmul(ROTATE_MATRIX, current)
    if token == "s":
        return _matmul(REFLECT_MATRIX, current)
    raise ValueError(f"Unknown D4 generator: {token}")


def _build_canonical_lookup():
    lookup = {}
    for element_name, tokens in D4_CANONICAL_SEQUENCES.items():
        matrix = IDENTITY_MATRIX
        for token in tokens:
            matrix = _apply_generator_to_matrix(matrix, token)
        lookup[matrix] = list(tokens)
    return lookup


CANONICAL_SEQUENCE_BY_MATRIX = _build_canonical_lookup()


def canonicalize_d4_sequence(sequence: List[str]) -> List[str]:
    matrix = IDENTITY_MATRIX
    for token in sequence:
        matrix = _apply_generator_to_matrix(matrix, token)
    return list(CANONICAL_SEQUENCE_BY_MATRIX[matrix])


def inverse_d4_sequence(sequence: List[str]) -> List[str]:
    inverse_sequence = []
    for token in reversed(sequence):
        if token == "e":
            inverse_sequence.append("e")
        elif token == "r":
            inverse_sequence.extend(["r", "r", "r"])
        elif token == "s":
            inverse_sequence.append("s")
        else:
            raise ValueError(f"Unknown D4 generator: {token}")
    return canonicalize_d4_sequence(inverse_sequence)


class ImageTransformer:
    """
    Applies D4 transformations and returns the transformed image plus target tokens.

    target_mode:
      - "applied": target is the applied transform g, so I2 = g(I1)
      - "completion": target is g^{-1}, so target(I2) is equivalent to I1
    """

    def __init__(self, allowed_elements: Optional[List[str]] = None, target_mode: str = "completion"):
        if allowed_elements is None:
            allowed_elements = list(D4_CANONICAL_SEQUENCES.keys())
        invalid = [name for name in allowed_elements if name not in D4_CANONICAL_SEQUENCES]
        if invalid:
            raise ValueError(f"Unknown D4 elements: {invalid}")
        if not allowed_elements:
            raise ValueError("allowed_elements must not be empty")
        if target_mode not in ("applied", "completion", "inverse"):
            raise ValueError("target_mode must be 'applied', 'completion', or 'inverse'")

        self.transformations = list(allowed_elements)
        self.target_mode = target_mode

    def apply_transform(self, image: Image.Image, transform: str) -> Image.Image:
        if transform == "e":
            return image
        elif transform == "r":
            return image.rotate(90, expand=True)
        elif transform == "s":
            return ImageOps.mirror(image)
        else:
            raise ValueError(f"Unknown D4 generator: {transform}")

    def sample_transformations(self, image: Optional[Image.Image] = None, p: Optional[float] = None) -> List[str]:
        del image, p
        element = random.choice(self.transformations)
        return list(D4_CANONICAL_SEQUENCES[element])

    def build_target_sequence(self, applied_sequence: List[str]) -> List[str]:
        if self.target_mode == "applied":
            return list(applied_sequence)
        return inverse_d4_sequence(applied_sequence)

    def transform(self, image: Image.Image, p: Optional[float] = None) -> Tuple[Image.Image, List[str]]:
        applied_sequence = self.sample_transformations(image, p)
        transformed_image = image.copy()
        for transform in applied_sequence:
            transformed_image = self.apply_transform(transformed_image, transform)
        target_sequence = self.build_target_sequence(applied_sequence)
        return transformed_image.convert('RGB'), target_sequence
