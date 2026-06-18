import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.mobilenet_embedding import MobileNetEmbedding
from app.utils.image_utils import get_inference_transforms
from app.utils.mapping_utils import save_id_mapping
from training.datasets import LandmarkDataset


def build_index(
    model_path: str,
    data_dir: str,
    embedding_dim: int = 128,
    batch_size: int = 32,
    output_dir: str = "artifacts",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    model = MobileNetEmbedding(embedding_dim=embedding_dim, pretrained=False)
    model_file = Path(model_path)

    if model_file.exists():
        state_dict = torch.load(str(model_file), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded model from {}", model_file)
    else:
        logger.warning("Model not found at {} — using pretrained weights", model_file)
        model = MobileNetEmbedding(embedding_dim=embedding_dim, pretrained=True)

    model = model.to(device)
    model.eval()

    transform = get_inference_transforms()
    dataset = LandmarkDataset(data_dir, transform=transform)
    logger.info("Dataset loaded | images={} | classes={}", len(dataset), dataset.num_classes)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_embeddings = []
    all_landmark_ids = []

    logger.info("Extracting embeddings...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Building embeddings"):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())

            for label in labels.numpy():
                landmark_id = dataset.get_class_name(label)
                all_landmark_ids.append(landmark_id)

    embeddings_array = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    id_array = np.array(all_landmark_ids)

    logger.info(
        "Embeddings extracted | shape={} | dtype={}",
        embeddings_array.shape, embeddings_array.dtype,
    )

    logger.info("Building FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)
    logger.info("Index built | vectors={} | dimension={}", index.ntotal, index.d)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_path = output_path / "landmarks.index"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved to {}", index_path)

    mapping_path = output_path / "id_mapping.npy"
    save_id_mapping(id_array, mapping_path)
    logger.info("ID mapping saved to {} | entries={}", mapping_path, len(id_array))

    logger.info("Verification: running test search...")
    test_query = embeddings_array[0:1]
    distances, indices = index.search(test_query, 5)
    logger.info("Test query results | distances={} | indices={}", distances[0], indices[0])
    logger.info("Index build complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from trained model embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/subset/train",
        help="Path to training data directory",
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
        default="artifacts",
        help="Directory to save FAISS index and mapping",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True, format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    ))

    build_index(
        model_path=args.model_path,
        data_dir=args.data_dir,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
