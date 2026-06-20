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
    landmark_name: str = "",
) -> Image.Image:
    rng = random.Random(hash(f"{landmark_id}_{image_idx}"))
    np_rng = np.random.RandomState(hash(f"{landmark_id}_{image_idx}") & 0xFFFFFFFF)

    golden_ratio = 0.618033988749895
    base_hue = (class_idx * golden_ratio) % 1.0

    # Per-image hue variation for diversity
    hue = (base_hue + rng.uniform(-0.08, 0.08)) % 1.0
    sat = 0.5 + rng.random() * 0.45
    val = 0.4 + rng.random() * 0.5

    r, g, b = _hsv_to_rgb(hue, sat, val)

    # Choose gradient type per image (not just per class)
    gradient_type = rng.choice(["diagonal", "radial", "horizontal", "vertical", "multi"])

    image = Image.new("RGB", (size, size))
    pixels = image.load()

    # Secondary color for gradient blending
    hue2 = (hue + rng.uniform(0.15, 0.4)) % 1.0
    r2, g2, b2 = _hsv_to_rgb(hue2, 0.5 + rng.random() * 0.4, 0.3 + rng.random() * 0.5)

    for y in range(size):
        for x in range(size):
            if gradient_type == "diagonal":
                t = (x + y) / (2 * size)
            elif gradient_type == "radial":
                cx, cy = size / 2, size / 2
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                t = min(dist / (size * 0.7), 1.0)
            elif gradient_type == "horizontal":
                t = x / size
            elif gradient_type == "vertical":
                t = y / size
            else:  # multi-tone
                t = (abs(x - size // 2) + abs(y - size // 2)) / size
                t = min(t, 1.0)

            pr = int((r * (1 - t) + r2 * t) * 255)
            pg = int((g * (1 - t) + g2 * t) * 255)
            pb = int((b * (1 - t) + b2 * t) * 255)
            pixels[x, y] = (
                max(0, min(255, pr)),
                max(0, min(255, pg)),
                max(0, min(255, pb)),
            )

    draw = ImageDraw.Draw(image)

    # Each image gets a unique combination of pattern elements
    pattern_type = (class_idx + image_idx) % 8
    offset_x = rng.randint(-25, 25)
    offset_y = rng.randint(-25, 25)
    scale_factor = rng.uniform(0.6, 1.4)

    shape_color = (
        min(255, int(r * 255) + rng.randint(40, 120)),
        min(255, int(g * 255) + rng.randint(40, 120)),
        min(255, int(b * 255) + rng.randint(40, 120)),
    )

    center_x = size // 2 + offset_x
    center_y = size // 2 + offset_y

    # Draw primary structural pattern
    if pattern_type == 0:  # Concentric circles
        for radius in range(int(15 * scale_factor), int(95 * scale_factor), int(12 * scale_factor)):
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                outline=shape_color, width=rng.randint(1, 4),
            )
    elif pattern_type == 1:  # Cross
        w = rng.randint(2, 6)
        draw.line([(center_x, 15), (center_x, size - 15)], fill=shape_color, width=w)
        draw.line([(15, center_y), (size - 15, center_y)], fill=shape_color, width=w)
    elif pattern_type == 2:  # Triangle
        s = int(60 * scale_factor)
        draw.polygon(
            [(center_x, center_y - s), (center_x - s, center_y + s),
             (center_x + s, center_y + s)],
            outline=shape_color, width=rng.randint(2, 5),
        )
    elif pattern_type == 3:  # Diamond
        s = int(55 * scale_factor)
        draw.polygon(
            [(center_x, center_y - s), (center_x + s, center_y),
             (center_x, center_y + s), (center_x - s, center_y)],
            outline=shape_color, width=rng.randint(2, 5),
        )
    elif pattern_type == 4:  # Grid
        spacing = rng.randint(15, 35)
        for i in range(offset_x % spacing, size, spacing):
            draw.line([(i, 0), (i, size)], fill=shape_color, width=1)
        for j in range(offset_y % spacing, size, spacing):
            draw.line([(0, j), (size, j)], fill=shape_color, width=1)
    elif pattern_type == 5:  # Star burst
        import math
        num_rays = rng.randint(6, 14)
        ray_len = int(70 * scale_factor)
        for i in range(num_rays):
            angle = 2 * math.pi * i / num_rays + rng.uniform(-0.1, 0.1)
            ex = center_x + int(ray_len * math.cos(angle))
            ey = center_y + int(ray_len * math.sin(angle))
            draw.line([(center_x, center_y), (ex, ey)], fill=shape_color, width=2)
    elif pattern_type == 6:  # Nested rectangles
        for r_offset in range(10, int(80 * scale_factor), int(18 * scale_factor)):
            draw.rectangle(
                [center_x - r_offset, center_y - r_offset,
                 center_x + r_offset, center_y + r_offset],
                outline=shape_color, width=2,
            )
    else:  # Dots / stipple
        num_dots = rng.randint(20, 60)
        for _ in range(num_dots):
            dx = rng.randint(10, size - 10)
            dy = rng.randint(10, size - 10)
            dr = rng.randint(2, 6)
            draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=shape_color)

    # Add secondary texture overlay (unique per class)
    secondary_type = class_idx % 4
    secondary_color = (
        min(255, shape_color[0] + 30),
        min(255, shape_color[1] + 30),
        min(255, shape_color[2] + 30),
        80,  # alpha for RGBA
    )

    if secondary_type == 0:  # Diagonal stripes
        stripe_w = 8 + class_idx % 6
        for i in range(-size, size * 2, stripe_w * 2):
            draw.line([(i, 0), (i + size, size)], fill=shape_color, width=1)
    elif secondary_type == 1:  # Dots pattern
        dot_spacing = 18 + class_idx % 8
        for dx in range(0, size, dot_spacing):
            for dy in range(0, size, dot_spacing):
                draw.point((dx, dy), fill=shape_color)
    elif secondary_type == 2:  # Corner accents
        accent_size = 30
        for cx, cy in [(0, 0), (size, 0), (0, size), (size, size)]:
            draw.arc([cx - accent_size, cy - accent_size,
                      cx + accent_size, cy + accent_size], 0, 360,
                     fill=shape_color, width=2)

    # Add text label overlay for semantic grounding
    if landmark_name and rng.random() > 0.3:
        try:
            font = ImageFont.load_default()
            label = landmark_name[:16]
            tx = rng.randint(8, max(10, size - len(label) * 7))
            ty = rng.randint(8, size - 20)
            # Text shadow for readability
            draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
        except Exception:
            pass

    # Apply noise with wider variance
    img_array = np.array(image)
    noise_level = rng.randint(8, 20)
    noise_matrix = np_rng.randint(-noise_level, noise_level + 1, img_array.shape, dtype=np.int16)
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
    min_images: int = 25,
    max_images: int = 35,
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
            image = generate_landmark_image(landmark_id, class_idx, img_idx, landmark_name=name)
            image_path = train_class_dir / f"{landmark_id}_{img_idx:04d}.jpg"
            image.save(image_path, "JPEG", quality=90)
            total_images += 1

        for img_idx in range(num_test):
            image = generate_landmark_image(landmark_id, class_idx, num_train + img_idx, landmark_name=name)
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
        "--min-images", type=int, default=25,
        help="Minimum images per class",
    )
    parser.add_argument(
        "--max-images", type=int, default=35,
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
