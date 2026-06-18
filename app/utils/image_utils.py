from io import BytesIO
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


INPUT_SIZE = 224


def get_inference_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(source: Union[str, Path, bytes, BytesIO]) -> Image.Image:
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path)
        elif isinstance(source, bytes):
            image = Image.open(BytesIO(source))
        elif isinstance(source, BytesIO):
            image = Image.open(source)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        return image.convert("RGB")

    except (OSError, IOError) as e:
        raise ValueError(f"Failed to decode image: {e}") from e


def preprocess_image(
    image: Image.Image,
    transform: transforms.Compose = None,
) -> torch.Tensor:
    if transform is None:
        transform = get_inference_transforms()

    tensor = transform(image)

    return tensor.unsqueeze(0)


def load_and_preprocess(
    source: Union[str, Path, bytes, BytesIO],
    transform: transforms.Compose = None,
) -> torch.Tensor:
    image = load_image(source)
    return preprocess_image(image, transform)
