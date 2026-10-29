"""
scripts/download_data.py
------------------------
Dataset Acquisition Script for AI-Powered Urban Expansion Prediction MCA Project.

Dataset: EuroSAT (RGB) - Sentinel-2 Satellite Imagery
Authors: Patrick Helber et al., German Research Center for Artificial Intelligence (DFKI)
Host: Zenodo Open Repository (Record 7711810)
Direct URL: https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
License: Open Data Commons Open Database License (ODbL)

This script downloads a sample of satellite images suitable for urban-area analysis:
- 50 Residential images (urban housing, neighborhoods, local streets)
- 50 Industrial images (factories, warehouses, commercial complexes)
Total: 100 satellite images saved to data/raw/
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from PIL import Image

DATASET_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "EuroSAT_RGB.zip")

SAMPLE_TARGETS = {
    "Residential": 50,
    "Industrial": 50
}


def download_archive(url: str, dest_path: str):
    """Downloads the EuroSAT zip archive if not already present."""
    if os.path.exists(dest_path):
        print(f"[INFO] Archive already exists: {dest_path}")
        return True

    print(f"[INFO] Downloading EuroSAT dataset from: {url}")
    print("[INFO] Please wait, downloading ~90 MB...")

    headers = {"User-Agent": "curl/8.0.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 512

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r[DOWNLOAD] {downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({percent:.1f}%)", end="", flush=True)

        print("\n[SUCCESS] Download completed successfully.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        print("\n--- ALTERNATIVE MANUAL INSTRUCTIONS ---")
        print("1. Open your browser and go to: https://zenodo.org/records/7711810")
        print("2. Download 'EuroSAT_RGB.zip' (90.3 MB).")
        print(f"3. Place the file inside: {RAW_DIR}")
        print("4. Re-run this script: python scripts/download_data.py")
        print("---------------------------------------")
        return False


def extract_sample_images(zip_path: str, output_dir: str):
    """Extracts a target number of urban images from the zip archive."""
    print(f"\n[INFO] Extracting urban satellite images to: {output_dir}")

    extracted_count = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        all_members = archive.namelist()

        for category, target_count in SAMPLE_TARGETS.items():
            category_prefix = f"EuroSAT_RGB/{category}/"
            category_files = [m for m in all_members if m.startswith(category_prefix) and m.lower().endswith((".jpg", ".jpeg", ".png"))]

            selected_files = category_files[:target_count]
            print(f"[INFO] Extracting {len(selected_files)} images from class '{category}'...")

            for member in selected_files:
                filename = os.path.basename(member)
                dest_file_path = os.path.join(output_dir, filename)

                # Extract file data and write directly into raw directory
                with archive.open(member) as source, open(dest_file_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                extracted_count += 1

    print(f"[SUCCESS] Extracted {extracted_count} urban satellite images.")
    return extracted_count


def validate_images(directory: str):
    """Verifies that all extracted images are valid and readable."""
    print(f"\n[INFO] Validating images in {directory}...")
    valid_count = 0
    invalid_count = 0

    image_files = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fname in image_files:
        fpath = os.path.join(directory, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()  # Verifies file integrity
            # Reopen to verify decode and dimensions
            with Image.open(fpath) as img:
                _ = img.size
                _ = img.mode
            valid_count += 1
        except Exception as e:
            print(f"[WARNING] Invalid image detected: {fname} ({e})")
            invalid_count += 1

    print(f"[VALIDATION REPORT] Total Checked: {len(image_files)} | Valid: {valid_count} | Corrupted/Invalid: {invalid_count}")
    return valid_count, invalid_count


def cleanup_archive(zip_path: str):
    """Removes the large zip archive after successful extraction to save space."""
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"[INFO] Cleaned up temporary archive: {os.path.basename(zip_path)}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. Download
    success = download_archive(DATASET_URL, ZIP_PATH)
    if not success:
        sys.exit(1)

    # 2. Extract sample
    extract_sample_images(ZIP_PATH, RAW_DIR)

    # 3. Clean up the large 90MB zip file
    cleanup_archive(ZIP_PATH)

    # 4. Validate
    valid_count, invalid_count = validate_images(RAW_DIR)

    if valid_count >= 50 and invalid_count == 0:
        print("\n[PHASE 2 COMPLETE] Dataset collected and verified successfully.")
    else:
        print("\n[PHASE 2 WARNING] Some images could not be validated. Check the log above.")


if __name__ == "__main__":
    main()
