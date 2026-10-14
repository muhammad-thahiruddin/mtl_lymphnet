"""Dataset inventory scanning for the LymphUs multicenter dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def build_dataset_inventory(data_dir: str) -> pd.DataFrame:
    """Scan the dataset folder structure and build a master DataFrame.

    Expected structure::

        data_dir/
            Center 1/
                Benign/    -> {id} Benign Image.bmp, {id} Benign Mask.tif, {id} Benign Lesion.bmp
                Malignant/ -> {id} Malignant Image.bmp, ...
            Center 2/
                Benign/
                Malignant/
    """
    records = []
    data_path = Path(data_dir)

    for center_dir in sorted(data_path.iterdir()):
        if not center_dir.is_dir() or "Center" not in center_dir.name:
            continue
        center_name = center_dir.name

        for class_dir in sorted(center_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name  # 'Benign' or 'Malignant'

            image_files = sorted(
                class_dir.glob(f"* {class_name} Image.bmp"),
                key=lambda p: int(p.name.split()[0]),
            )

            for img_path in image_files:
                sample_id = int(img_path.name.split()[0])
                prefix = f"{sample_id} {class_name}"

                mask_path = class_dir / f"{prefix} Mask.tif"
                lesion_path = class_dir / f"{prefix} Lesion.bmp"

                records.append(
                    {
                        "sample_id": sample_id,
                        "center": center_name,
                        "label_name": class_name,
                        "label": 0 if class_name == "Benign" else 1,
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "lesion_path": str(lesion_path),
                        "mask_exists": mask_path.exists(),
                        "lesion_exists": lesion_path.exists(),
                    }
                )

    return pd.DataFrame(records)


def validate_inventory(df: pd.DataFrame, verify_readable: bool = True) -> dict:
    """Validate completeness (missing masks/lesions) and image readability.

    Returns a summary dict; raises no exceptions so callers can decide how to
    handle incomplete data.
    """
    n_missing_mask = int((~df["mask_exists"]).sum())
    n_missing_lesion = int((~df["lesion_exists"]).sum())

    corrupt = []
    if verify_readable:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
            try:
                img = Image.open(row["image_path"])
                img.verify()
            except Exception as e:  # noqa: BLE001 - report, don't crash
                corrupt.append((row["center"], row["label_name"], row["sample_id"], str(e)))

    return {
        "n_total": len(df),
        "n_missing_mask": n_missing_mask,
        "n_missing_lesion": n_missing_lesion,
        "corrupt_images": corrupt,
    }
