from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from app.models.mobilenet_embedding import MobileNetEmbedding
from app.models.triplet_loss import TripletLoss
from training.datasets import TripletDataset, LandmarkDataset
from training.samplers import BalancedBatchSampler
from training.hard_negative_mining import mine_hard_triplets


class Trainer:
    def __init__(
        self,
        model: MobileNetEmbedding,
        train_dir: str,
        strategy: str = "random",
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        margin: float = 0.3,
        embedding_dim: int = 128,
        save_dir: str = "artifacts",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_dir = train_dir
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.save_dir = Path(save_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        self.criterion = TripletLoss(margin=margin)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=learning_rate * 0.01,
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float("inf")

        logger.info(
            "Trainer initialized | strategy={} | epochs={} | batch_size={} | "
            "lr={} | margin={} | device={}",
            strategy, epochs, batch_size, learning_rate, margin, self.device,
        )

    def train(self) -> dict:
        if self.strategy == "random":
            return self._train_random()
        elif self.strategy == "hard":
            return self._train_hard()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'random' or 'hard'.")

    def _train_random(self) -> dict:
        logger.info("Starting random mining training")

        from app.utils.image_utils import get_training_transforms
        dataset = TripletDataset(
            root_dir=self.train_dir,
            transform=get_training_transforms(),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True,
        )

        history = {"epoch_losses": []}

        for epoch in range(self.epochs):
            epoch_loss = self._train_epoch_random(dataloader, epoch)
            history["epoch_losses"].append(epoch_loss)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            logger.info(
                "Epoch {}/{} | loss={:.4f} | lr={:.6f}",
                epoch + 1, self.epochs, epoch_loss, current_lr,
            )

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint("model.pth")
                logger.info("New best model saved | loss={:.4f}", epoch_loss)

        return history

    def _train_epoch_random(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [random]",
            leave=False,
        )

        for anchor, positive, negative in progress:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)

            loss = self.criterion(anchor_emb, positive_emb, negative_emb)

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(num_batches, 1)

    def _train_hard(self) -> dict:
        logger.info("Starting hard mining training")

        from app.utils.image_utils import get_training_transforms
        dataset = LandmarkDataset(
            root_dir=self.train_dir,
            transform=get_training_transforms(),
        )

        p_classes = min(8, dataset.num_classes)
        k_samples = max(2, self.batch_size // p_classes)

        sampler = BalancedBatchSampler(
            class_to_samples=dataset.class_to_samples,
            p_classes=p_classes,
            k_samples=k_samples,
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        history = {"epoch_losses": []}

        for epoch in range(self.epochs):
            epoch_loss = self._train_epoch_hard(dataloader, epoch)
            history["epoch_losses"].append(epoch_loss)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            logger.info(
                "Epoch {}/{} | loss={:.4f} | lr={:.6f}",
                epoch + 1, self.epochs, epoch_loss, current_lr,
            )

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint("model.pth")
                logger.info("New best model saved | loss={:.4f}", epoch_loss)

        return history

    def _train_epoch_hard(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [hard]",
            leave=False,
        )

        for images, labels in progress:
            images = images.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.model(images)

            anchors, hard_positives, hard_negatives = mine_hard_triplets(
                embeddings, labels, margin=self.margin,
            )

            if anchors.size(0) == 0:
                continue

            loss = self.criterion(anchors, hard_positives, hard_negatives)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, filename: str) -> None:
        save_path = self.save_dir / filename
        torch.save(self.model.state_dict(), save_path)
        logger.debug("Checkpoint saved to {}", save_path)
