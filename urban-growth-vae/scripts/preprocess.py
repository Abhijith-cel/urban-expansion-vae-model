"""
scripts/preprocess.py
---------------------
Image Preprocessing Pipeline for Urban Expansion VAE Project.

This script:
1. Reads raw satellite images from data/raw/
2. Skips invalid or unreadable images
3. Converts each image to RGB format
4. Resizes to 128 x 128 pixels
5. Normalizes pixel values to [0.0, 1.0] for ML modeling
6. Saves processed images into data/processed/
7. Prints a summary of found, processed, and skipped images
8. Displays an original vs processed visual comparison using Matplotlib
"""

import os
import sys
import argparse
import tempfile
import numpy as np
from PIL import Image

# Ensure Matplotlib has a writable cache directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib.pyplot as plt

# Define project directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

TARGET_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_and_preprocess_image(image_path: str):
    """
    Reads an image, converts it to RGB, resizes to target size,
    and returns both the resized PIL Image and normalized numpy array [0, 1].
    """
    # Open and verify image
    with Image.open(image_path) as img:
        # Convert to RGB (handles grayscale, RGBA, CMYK, etc.)
        img_rgb = img.convert("RGB")
        
        # Resize to 128x128
        img_resized = img_rgb.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        
        # Normalize pixel values to 0.0 - 1.0 range
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

    return img_rgb, img_resized, img_array


def display_comparison(original_img: Image.Image, processed_array: np.ndarray, save_path: str = None, show_window: bool = True):
    """
    Displays a side-by-side comparison of the original image and its processed version.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Original Image
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original Image\nDimensions: {original_img.size[0]}x{original_img.size[1]} | Mode: {original_img.mode}")
    axes[0].axis("off")

    # Processed Image
    axes[1].imshow(processed_array)
    axes[1].set_title(f"Processed for VAE\nDimensions: 128x128 | Range: [{processed_array.min():.2f}, {processed_array.max():.2f}]")
    axes[1].axis("off")

    plt.suptitle("Urban Satellite Image Preprocessing Verification", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Comparison figure saved to: {save_path}")

    if show_window:
        print("[INFO] Displaying comparison window (close the window to finish)...")
        plt.show()
    else:
        plt.close()


def preprocess_all(show_display: bool = True):
    """
    Processes all raw images, saves them to data/processed/, and displays a sample comparison.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_DIR):
        print(f"[ERROR] Raw directory not found: {RAW_DIR}")
        print("Please run 'python scripts/download_data.py' first.")
        return

    # Find candidate image files
    all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    total_found = len(all_files)

    if total_found == 0:
        print(f"[WARNING] No images found in {RAW_DIR}.")
        print("Please run 'python scripts/download_data.py' to collect dataset images.")
        return

    print(f"[INFO] Found {total_found} image(s) in {RAW_DIR}. Starting preprocessing...")

    processed_count = 0
    skipped_count = 0
    sample_for_comparison = None

    for fname in all_files:
        src_path = os.path.join(RAW_DIR, fname)
        base_name, _ = os.path.splitext(fname)
        dst_path = os.path.join(PROCESSED_DIR, f"{base_name}.png")

        try:
            original_rgb, processed_pil, processed_norm = load_and_preprocess_image(src_path)

            # Save the clean 128x128 RGB image
            processed_pil.save(dst_path, format="PNG")
            processed_count += 1

            # Retain the first successfully processed image for visual comparison
            if sample_for_comparison is None:
                sample_for_comparison = (original_rgb, processed_norm)

        except Exception as err:
            print(f"[SKIP] Failed to process '{fname}': {err}")
            skipped_count += 1

    # Print summary report
    print("\n" + "=" * 45)
    print("       PREPROCESSING SUMMARY REPORT")
    print("=" * 45)
    print(f"Total raw images found      : {total_found}")
    print(f"Successfully processed      : {processed_count}")
    print(f"Skipped / Invalid files     : {skipped_count}")
    print(f"Processed images directory  : {PROCESSED_DIR}")
    print("=" * 45 + "\n")

    # Display comparison for the sample image
    if sample_for_comparison is not None:
        orig_img, proc_arr = sample_for_comparison
        comparison_plot_file = os.path.join(PROCESSED_DIR, "comparison.png")
        display_comparison(orig_img, proc_arr, save_path=comparison_plot_file, show_window=show_display)


def main():
    parser = argparse.ArgumentParser(description="Preprocess satellite images for Urban Growth VAE.")
    parser.add_argument("--no-display", action="store_true", help="Save the comparison plot without opening a GUI window")
    args = parser.parse_args()

    preprocess_all(show_display=not args.no_display)


if __name__ == "__main__":
    main()
