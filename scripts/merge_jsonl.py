#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atomically merge JSONL files")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [path for path in args.inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing JSONL inputs: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            for path in args.inputs:
                with path.open("r", encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            target.write(line if line.endswith("\n") else line + "\n")
        os.replace(temporary_name, args.output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    print(f"Merged {len(args.inputs)} files into {args.output}")


if __name__ == "__main__":
    main()
