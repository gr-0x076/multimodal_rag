"""
ContextMesh — Pipeline Data Reset Utility

Cleans all generated artifacts (evidence.json, extracted frames, transcripts)
from data/processed/ to prepare for a clean, fresh video ingestion run.

Usage:
    python pipeline/reset.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
processed_dir = os.path.join(root_dir, "data", "processed")


def _force_remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree to remove read-only or locked files."""
    try:
        os.chmod(path, 0o777)
        func(path)
    except Exception:
        pass


def reset_processed_data(verbose: bool = True) -> None:
    """Removes all generated evidence files, transcripts, and frames."""
    if not os.path.exists(processed_dir):
        if verbose:
            print(f"Nothing to reset. {processed_dir} does not exist.")
        return

    removed_files = 0
    removed_dirs = 0

    for item in os.listdir(processed_dir):
        item_path = os.path.join(processed_dir, item)
        if item == ".gitkeep":
            continue
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, onerror=_force_remove_readonly)
                removed_dirs += 1
            else:
                try:
                    os.remove(item_path)
                except Exception:
                    os.chmod(item_path, 0o777)
                    os.remove(item_path)
                removed_files += 1
        except Exception as e:
            if verbose:
                print(f"Notice: {item}: {e}")

    if verbose:
        print("\n=======================================================")
        print(" ContextMesh Pipeline Data Reset")
        print("=======================================================")
        print(f" Cleaned: {processed_dir}")
        print(f" Preserved: Source videos in test_data/ & data/raw/")
        print(" Ready for clean video ingestion!")
        print("=======================================================\n")


if __name__ == "__main__":
    reset_processed_data(verbose=True)
