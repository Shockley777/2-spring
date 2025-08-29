#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import List, Set


def find_files(root: Path, patterns: List[str]) -> Set[Path]:
    matches: Set[Path] = set()
    for pattern in patterns:
        for p in root.rglob(pattern):
            if p.is_file():
                matches.add(p)
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Delete Cellpose visualization files such as *_flows.png (optionally *_outlines.png)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Root directory to search from. Default: DATASET6/",
    )
    parser.add_argument(
        "--include-outlines",
        action="store_true",
        help="Also delete *_outlines.png files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files to be deleted without actually deleting them.",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    if not root.exists():
        print(f"Root path does not exist: {root}")
        return

    patterns = ["*_flows.png"]
    if args.include_outlines:
        patterns.append("*_outlines.png")

    files = sorted(find_files(root, patterns))

    if not files:
        print("No matching files found.")
        return

    print(f"Found {len(files)} files under {root}:")
    for f in files:
        print(f" - {f}")

    if args.dry_run:
        print("\nDry run: no files were deleted.")
        return

    deleted = 0
    for f in files:
        try:
            f.unlink(missing_ok=True)
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    print(f"\nDeleted {deleted} files.")


if __name__ == "__main__":
    main()


