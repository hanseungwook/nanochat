"""Resumable CLI for the pinned Nemotron Specialized packed corpus."""

from __future__ import annotations

import argparse
import json
import os

from nanochat.nemotron_data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MIN_FREE_GB,
    DEFAULT_SHARD_GB,
    data_layout,
    initialize_layout,
    reject_repository_local_root,
    run_audit,
    run_pack,
    run_shuffle,
    run_tokenize,
    run_verify,
)


def _default_data_root() -> str:
    return os.environ.get("NANOCHAT_DATA_ROOT", str(DEFAULT_DATA_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_nemotron",
        description="Build and verify the pinned Nemotron Specialized packed corpus",
    )
    parser.add_argument("--data-root", default=_default_data_root(), help="persistent data root")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help="minimum free space required before a bulk stage",
    )
    parser.add_argument("--job-index", type=int, default=0, help="zero-based job-array index")
    parser.add_argument("--job-count", type=int, default=1, help="total job-array task count")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="pin artifacts, mirror raw files, and audit sources")
    audit.add_argument("--sample-per-source", type=int, default=1024)
    audit.add_argument("--selection-oversample", type=float, default=1.05)
    audit.add_argument("--skip-mirror", action="store_true", help="audit an already complete raw mirror")
    audit.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="download and verify only the pinned tokenizer artifacts",
    )

    tokenize = subparsers.add_parser("tokenize", help="offline-tokenize selected documents")
    tokenize.add_argument("--tokenizer-batch-size", type=int, default=128)

    pack = subparsers.add_parser("pack", help="build exact source-pure sequence pools")
    pack.add_argument("--buffer-size", type=int, default=1024)
    pack.add_argument("--shard-size-gb", type=float, default=DEFAULT_SHARD_GB)

    shuffle = subparsers.add_parser("shuffle", help="materialize shuffled segments and manifest")
    shuffle.add_argument("--shard-size-gb", type=float, default=DEFAULT_SHARD_GB)

    verify = subparsers.add_parser("verify", help="fully verify final files, quotas, and hashes")
    verify.add_argument(
        "--trust-tokenizer-files",
        action="store_true",
        help="skip re-hashing tokenizer files (manifest metadata is still checked)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.job_count <= 0 or not 0 <= args.job_index < args.job_count:
        parser.error("--job-index must be in [0, --job-count)")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be non-negative")
    reject_repository_local_root(args.data_root)
    layout = data_layout(args.data_root)
    old_umask = os.umask(0o002)
    try:
        initialize_layout(layout)
        resolved = {
            "data_root": str(layout.root),
            "raw": str(layout.raw),
            "staging": str(layout.staging),
            "packed": str(layout.packed),
            "tokenizer": str(layout.tokenizer),
            "runtime": str(layout.runtime),
        }
        print(json.dumps({"event": "resolved_paths", "paths": resolved}, sort_keys=True), flush=True)
        if args.command == "audit":
            result = run_audit(
                layout,
                job_index=args.job_index,
                job_count=args.job_count,
                sample_per_source=args.sample_per_source,
                oversample=args.selection_oversample,
                skip_mirror=args.skip_mirror,
                tokenizer_only=args.tokenizer_only,
                min_free_gb=args.min_free_gb,
            )
        elif args.command == "tokenize":
            result = run_tokenize(
                layout,
                job_index=args.job_index,
                job_count=args.job_count,
                batch_size=args.tokenizer_batch_size,
                min_free_gb=args.min_free_gb,
            )
        elif args.command == "pack":
            result = run_pack(
                layout,
                job_index=args.job_index,
                job_count=args.job_count,
                buffer_size=args.buffer_size,
                shard_gb=args.shard_size_gb,
                min_free_gb=args.min_free_gb,
            )
        elif args.command == "shuffle":
            result = run_shuffle(
                layout,
                job_index=args.job_index,
                job_count=args.job_count,
                shard_gb=args.shard_size_gb,
                min_free_gb=args.min_free_gb,
            )
        elif args.command == "verify":
            result = run_verify(layout, tokenizer_hash_files=not args.trust_tokenizer_files)
        else:  # pragma: no cover - argparse enforces this
            raise AssertionError(args.command)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
