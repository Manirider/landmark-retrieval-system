import argparse
import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.mobilenet_embedding import MobileNetEmbedding
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the landmark embedding model with metric learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "hard"],
        help="Mining strategy: 'random' for offline triplets, 'hard' for online hard mining",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding vector dimension",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Triplet loss margin",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/subset/train",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="artifacts",
        help="Directory to save model checkpoints",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True, format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    ))

    logger.info("=" * 60)
    logger.info("Landmark Retrieval — Training")
    logger.info("=" * 60)
    logger.info("Strategy:      {}", args.strategy)
    logger.info("Epochs:        {}", args.epochs)
    logger.info("Batch size:    {}", args.batch_size)
    logger.info("Embedding dim: {}", args.embedding_dim)
    logger.info("Learning rate: {}", args.learning_rate)
    logger.info("Margin:        {}", args.margin)
    logger.info("Train dir:     {}", args.train_dir)
    logger.info("Save dir:      {}", args.save_dir)
    logger.info("=" * 60)

    model = MobileNetEmbedding(
        embedding_dim=args.embedding_dim,
        pretrained=True,
    )

    trainer = Trainer(
        model=model,
        train_dir=args.train_dir,
        strategy=args.strategy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        embedding_dim=args.embedding_dim,
        save_dir=args.save_dir,
    )

    history = trainer.train()

    final_loss = history["epoch_losses"][-1] if history["epoch_losses"] else float("inf")
    best_loss = min(history["epoch_losses"]) if history["epoch_losses"] else float("inf")

    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("Final loss: {:.4f}", final_loss)
    logger.info("Best loss:  {:.4f}", best_loss)
    logger.info("Model saved to: {}", Path(args.save_dir) / "model.pth")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
