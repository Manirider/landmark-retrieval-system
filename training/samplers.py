import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler
from loguru import logger


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        class_to_samples: Dict[int, List[int]],
        p_classes: int = 8,
        k_samples: int = 4,
    ) -> None:
        self.class_to_samples = {
            cls: samples
            for cls, samples in class_to_samples.items()
            if len(samples) >= k_samples
        }

        self.p_classes = min(p_classes, len(self.class_to_samples))
        self.k_samples = k_samples
        self.batch_size = self.p_classes * self.k_samples

        self.available_classes = list(self.class_to_samples.keys())

        if self.p_classes < 2:
            raise ValueError(
                f"Need at least 2 classes with {k_samples}+ samples each "
                f"for balanced batch sampling. Found {len(self.class_to_samples)}."
            )

        total_samples = sum(len(s) for s in self.class_to_samples.values())
        self.num_batches = total_samples // self.batch_size

        logger.info(
            "BalancedBatchSampler | P={} classes | K={} samples | "
            "batch_size={} | batches/epoch={}",
            self.p_classes, self.k_samples, self.batch_size, self.num_batches,
        )

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch_indices = []

            selected_classes = random.sample(self.available_classes, self.p_classes)

            for cls in selected_classes:
                samples = self.class_to_samples[cls]

                if len(samples) >= self.k_samples:
                    chosen = random.sample(samples, self.k_samples)
                else:
                    chosen = random.choices(samples, k=self.k_samples)

                batch_indices.extend(chosen)

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches
