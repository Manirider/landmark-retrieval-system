import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger


def load_landmark_map(path: Path) -> Dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark map not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        if not isinstance(mapping, dict):
            raise ValueError(f"Expected dict, got {type(mapping).__name__}")

        logger.debug("Loaded landmark map with {} entries", len(mapping))
        return mapping

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse landmark map JSON: {e}") from e


def save_landmark_map(mapping: Dict[str, str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    logger.debug("Saved landmark map with {} entries to {}", len(mapping), path)


def load_id_mapping(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ID mapping not found: {path}")

    mapping = np.load(str(path), allow_pickle=True)
    logger.debug("Loaded ID mapping with {} entries", len(mapping))
    return mapping


def save_id_mapping(mapping: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(path), mapping)
    logger.debug("Saved ID mapping with {} entries to {}", len(mapping), path)


def resolve_landmark_names(
    landmark_ids: List[str],
    landmark_map: Dict[str, str],
) -> List[str]:
    names = []
    for lid in landmark_ids:
        name = landmark_map.get(str(lid), f"Unknown Landmark ({lid})")
        names.append(name)
    return names
