#!/usr/bin/env python3
"""
Convert HEIC images to PNG format for training data preparation.
"""

import os
from pathlib import Path
from PIL import Image
import pillow_heif


def convert_heic_to_png(input_dir, output_dir=None):
    """
    Convert all HEIC files in input_dir to PNG format.

    Args:
        input_dir (str): Directory containing HEIC files
        output_dir (str): Directory to save PNG files (optional, defaults to input_dir)
    """
    # Register HEIF opener with Pillow
    pillow_heif.register_heif_opener()

    input_path = Path(input_dir)

    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    heic_files = list(input_path.glob("*.HEIC")) + list(input_path.glob("*.heic"))

    if not heic_files:
        print(f"No HEIC files found in {input_dir}")
        return

    print(f"Found {len(heic_files)} HEIC files to convert...")

    converted_count = 0
    for heic_file in heic_files:
        try:
            with Image.open(heic_file) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                png_filename = heic_file.stem + ".png"
                png_path = output_path / png_filename

                img.save(png_path, "PNG", optimize=True)
                print(f"Converted: {heic_file.name} -> {png_filename}")
                converted_count += 1

        except Exception as e:
            print(f"Error converting {heic_file.name}: {e}")

    print(
        f"\nConversion complete! {converted_count}/{len(heic_files)} files converted successfully."
    )


def remove_heic_files(directory):
    """
    Remove all HEIC files from the specified directory.

    Args:
        directory (str): Directory containing HEIC files to remove
    """
    dir_path = Path(directory)

    heic_files = list(dir_path.glob("*.HEIC")) + list(dir_path.glob("*.heic"))

    if not heic_files:
        print(f"No HEIC files found to remove in {directory}")
        return

    print(f"Removing {len(heic_files)} HEIC files...")

    removed_count = 0
    for heic_file in heic_files:
        try:
            heic_file.unlink()
            print(f"Removed: {heic_file.name}")
            removed_count += 1
        except Exception as e:
            print(f"Error removing {heic_file.name}: {e}")

    print(
        f"Removal complete! {removed_count}/{len(heic_files)} HEIC files removed successfully."
    )


if __name__ == "__main__":
    train_raw_dir = "data/train/raw"
    convert_heic_to_png(train_raw_dir)

    print("\nRemoving original HEIC files...")
    remove_heic_files(train_raw_dir)

    valid_raw_dir = "data/valid/raw"
    if os.path.exists(valid_raw_dir):
        convert_heic_to_png(valid_raw_dir)
        remove_heic_files(valid_raw_dir)

    remove_heic_files(train_raw_dir)
    if os.path.exists(valid_raw_dir):
        remove_heic_files(valid_raw_dir)
