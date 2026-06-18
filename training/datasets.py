import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from loguru import logger

from app.utils.image_utils import get_training_transforms, get_inference_transforms


class LandmarkDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform or get_inference_transforms()

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_samples: Dict[int, List[int]] = {}

        self._scan_directory()

    def _scan_directory(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.root_dir}")

        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            self.class_to_samples[class_idx] = []

            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            image_files = sorted([
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ])

            for img_path in image_files:
                sample_idx = len(self.samples)
                self.samples.append((img_path, class_idx))
                self.class_to_samples[class_idx].append(sample_idx)

        logger.info(
            "Dataset loaded | classes={} | samples={} | root={}",
            len(self.class_to_idx), len(self.samples), self.root_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, IOError) as e:
            logger.warning("Failed to load {}: {} — returning black image", img_path, e)
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        image_tensor = self.transform(image)
        return image_tensor, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def get_class_name(self, class_idx: int) -> str:
        return self.idx_to_class.get(class_idx, "unknown")


class TripletDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.base_dataset = LandmarkDataset(root_dir, transform or get_training_transforms())
        self.transform = self.base_dataset.transform

        self.valid_classes = [
            cls_idx for cls_idx, samples in self.base_dataset.class_to_samples.items()
            if len(samples) >= 2
        ]

        if len(self.valid_classes) < 2:
            raise ValueError(
                "Need at least 2 classes with 2+ images each for triplet training. "
                f"Found {len(self.valid_classes)} valid classes."
            )

        self.valid_samples = []
        for cls_idx in self.valid_classes:
            self.valid_samples.extend(self.base_dataset.class_to_samples[cls_idx])

        logger.info(
            "TripletDataset ready | valid_classes={} | valid_samples={}",
            len(self.valid_classes), len(self.valid_samples),
        )

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx = self.valid_samples[idx]
        anchor_img, anchor_label = self.base_dataset[anchor_idx]

        positive_candidates = [
            s for s in self.base_dataset.class_to_samples[anchor_label]
            if s != anchor_idx
        ]
        positive_idx = random.choice(positive_candidates)
        positive_img, _ = self.base_dataset[positive_idx]

        negative_classes = [c for c in self.valid_classes if c != anchor_label]
        negative_class = random.choice(negative_classes)
        negative_idx = random.choice(self.base_dataset.class_to_samples[negative_class])
        negative_img, _ = self.base_dataset[negative_idx]

        return anchor_img, positive_img, negative_img

    @property
    def num_classes(self) -> int:
        return self.base_dataset.num_classes
