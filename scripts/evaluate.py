import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.mobilenet_embedding import LandmarkEmbedding, BACKBONE_REGISTRY, MobileNetEmbedding
from app.utils.metrics import evaluate_retrieval
from app.utils.image_utils import get_inference_transforms
from training.datasets import LandmarkDataset


def extract_all_embeddings(
    model: MobileNetEmbedding,
    dataset: LandmarkDataset,
    device: torch.device,
    batch_size: int = 32,
) -> tuple:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_embeddings = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance with Recall@K metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/subset/test",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/subset/train",
        help="Path to training data directory (used as gallery)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "hard"],
        help="Which strategy's model to evaluate (for report naming)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation reports",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b3",
        choices=list(BACKBONE_REGISTRY.keys()),
        help="Backbone architecture to use",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True, format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)

    model = LandmarkEmbedding(embedding_dim=args.embedding_dim, pretrained=False, backbone=args.backbone)
    model_path = Path(args.model_path)

    if model_path.exists():
        state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded model from {}", model_path)
    else:
        logger.warning("Model not found at {} — using random weights", model_path)

    model = model.to(device)
    model.eval()

    transform = get_inference_transforms()
    test_dataset = LandmarkDataset(args.test_dir, transform=transform)
    train_dataset = LandmarkDataset(args.train_dir, transform=transform)

    logger.info("Test set: {} images | Train/Gallery: {} images",
                len(test_dataset), len(train_dataset))

    logger.info("Extracting test embeddings...")
    test_embeddings, test_labels = extract_all_embeddings(
        model, test_dataset, device, args.batch_size,
    )

    logger.info("Extracting gallery embeddings...")
    gallery_embeddings, gallery_labels = extract_all_embeddings(
        model, train_dataset, device, args.batch_size,
    )

    logger.info("Computing Recall@K metrics...")
    metrics = evaluate_retrieval(
        query_embeddings=test_embeddings,
        query_labels=test_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        k_values=[1, 5, 10],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{args.strategy}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 50)
    logger.info("Evaluation Results ({})", args.strategy)
    logger.info("=" * 50)
    for metric, value in metrics.items():
        logger.info("{}: {:.4f}", metric, value)
    logger.info("Report saved to: {}", report_path)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
