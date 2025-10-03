"""
Helper functions for working with filesystem
"""
from pathlib import Path


def all_filepaths_exist(
        filepaths: list[Path],
) -> bool:
    for filepath in filepaths:
        if not filepath.exists():
            return False
    return True