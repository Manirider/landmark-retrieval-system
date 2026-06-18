import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


LANDMARK_NAMES = [
    "Eiffel Tower", "Statue of Liberty", "Taj Mahal", "Great Wall of China",
    "Colosseum", "Machu Picchu", "Christ the Redeemer", "Petra",
    "Great Pyramid of Giza", "Angkor Wat", "Sagrada Familia",
    "Big Ben", "Sydney Opera House", "Golden Gate Bridge",
    "Burj Khalifa", "Tower of Pisa", "Hagia Sophia", "Stonehenge",
    "Mount Rushmore", "Neuschwanstein Castle", "Chichen Itza",
    "Alhambra", "Acropolis", "Notre Dame Cathedral", "Windsor Castle",
    "Empire State Building", "Brandenburg Gate", "Arc de Triomphe",
    "St. Basil's Cathedral", "Forbidden City", "Temple of Heaven",
    "Blue Mosque", "Dome of the Rock", "Tower Bridge",
    "Charles Bridge", "Rialto Bridge", "Brooklyn Bridge",
    "Palace of Versailles", "Buckingham Palace", "Topkapi Palace",
    "Lincoln Memorial", "Washington Monument", "Space Needle",
    "CN Tower", "Tokyo Tower", "Pantheon", "Trevi Fountain",
    "Spanish Steps", "Piazza San Marco", "Duomo di Milano",
    "La Pedrera", "Park Guell", "Mont Saint Michel",
    "Edinburgh Castle", "Prague Castle", "Hohenzollern Castle",
    "Sistine Chapel", "Vatican City", "Reichstag Building",
    "Opera Garnier", "Louvre Museum", "British Museum",
    "Hermitage Museum", "Metropolitan Museum", "Guggenheim Museum",
]


def generate_landmark_image(
    landmark_id: str,
    class_idx: int,
    image_idx: int,
    size: int = 224,
) -> Image.Image:
    rng = random.Random(hash(f"{landmark_id}_{image_idx}"))

    golden_ratio = 0.618033988749895
    hue = (class_idx * golden_ratio) % 1.0

    r, g, b = _hsv_to_rgb(hue, 0.7 + rng.random() * 0.2, 0.6 + rng.random() * 0.3)

    image = Image.new("RGB", (size, size))
    pixels = image.load()

    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size)
            pr = int(r * (1 - t) * 255 + r * 0.3 * t * 255)
            pg = int(g * (1 - t) * 255 + g * 0.5 * t * 255)
            pb = int(b * t * 255 + b * 0.7 * (1 - t) * 255)
            pixels[x, y] = (
                max(0, min(255, pr)),
                max(0, min(255, pg)),
                max(0, min(255, pb)),
            )

    draw = ImageDraw.Draw(image)

    pattern_type = class_idx % 5

    offset_x = rng.randint(-15, 15)
    offset_y = rng.randint(-15, 15)

    shape_color = (
        min(255, int(r * 255) + 80),
        min(255, int(g * 255) + 80),
        min(255, int(b * 255) + 80),
    )

    center_x = size // 2 + offset_x
    center_y = size // 2 + offset_y

    if pattern_type == 0:
        for radius in range(20, 90, 15):
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                outline=shape_color, width=2,
            )
    elif pattern_type == 1:
        draw.line([(center_x, 20), (center_x, size - 20)], fill=shape_color, width=4)
        draw.line([(20, center_y), (size - 20, center_y)], fill=shape_color, width=4)
    elif pattern_type == 2:
        draw.polygon(
            [(center_x, 30), (30 + offset_x, size - 30), (size - 30 + offset_x, size - 30)],
            outline=shape_color, width=3,
        )
    elif pattern_type == 3:
        draw.polygon(
            [(center_x, 30), (size - 30, center_y), (center_x, size - 30), (30, center_y)],
            outline=shape_color, width=3,
        )
    else:
        for i in range(30 + offset_x % 10, size - 20, 25):
            draw.line([(i, 20), (i, size - 20)], fill=shape_color, width=1)
        for j in range(30 + offset_y % 10, size - 20, 25):
            draw.line([(20, j), (size - 20, j)], fill=shape_color, width=1)

    noise = np.random.RandomState(hash(f"{landmark_id}_{image_idx}") & 0xFFFFFFFF)
    img_array = np.array(image)
    noise_matrix = noise.randint(-10, 11, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise_matrix, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    return [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i]


def create_subset(
    num_classes: int = 50,
    min_images: int = 10,
    max_images: int = 15,
    train_split: float = 0.8,
    output_dir: str = "data",
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    subset_dir = output_dir / "subset"
    train_dir = subset_dir / "train"
    test_dir = subset_dir / "test"

    if subset_dir.exists():
        import shutil
        shutil.rmtree(subset_dir)
        logger.info("Cleaned existing subset directory")

    num_classes = min(num_classes, len(LANDMARK_NAMES))
    selected_landmarks = LANDMARK_NAMES[:num_classes]

    landmark_map = {}
    total_images = 0

    logger.info(
        "Creating subset | classes={} | images_per_class={}-{} | split={:.0%}",
        num_classes, min_images, max_images, train_split,
    )

    for class_idx, name in enumerate(tqdm(selected_landmarks, desc="Generating classes")):
        landmark_id = str(10000 + class_idx)
        landmark_map[landmark_id] = name

        num_images = random.randint(min_images, max_images)
        num_train = max(2, int(num_images * train_split))
        num_test = max(2, num_images - num_train)

        train_class_dir = train_dir / landmark_id
        test_class_dir = test_dir / landmark_id
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(num_train):
            image = generate_landmark_image(landmark_id, class_idx, img_idx)
            image_path = train_class_dir / f"{landmark_id}_{img_idx:04d}.jpg"
            image.save(image_path, "JPEG", quality=90)
            total_images += 1

        for img_idx in range(num_test):
            image = generate_landmark_image(landmark_id, class_idx, num_train + img_idx)
            image_path = test_class_dir / f"{landmark_id}_{num_train + img_idx:04d}.jpg"
            image.save(image_path, "JPEG", quality=90)
            total_images += 1

    map_path = output_dir / "landmark_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(landmark_map, f, indent=2, ensure_ascii=False)

    logger.info(
        "Subset created | classes={} | total_images={} | map={}",
        num_classes, total_images, map_path,
    )
    logger.info("Train directory: {}", train_dir)
    logger.info("Test directory: {}", test_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced landmark image subset for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-classes", type=int, default=50,
        help="Number of landmark classes",
    )
    parser.add_argument(
        "--min-images", type=int, default=10,
        help="Minimum images per class",
    )
    parser.add_argument(
        "--max-images", type=int, default=15,
        help="Maximum images per class",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8,
        help="Fraction of images for training",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Root output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)

    create_subset(
        num_classes=args.num_classes,
        min_images=args.min_images,
        max_images=args.max_images,
        train_split=args.train_split,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
