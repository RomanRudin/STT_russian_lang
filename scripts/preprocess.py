#!/usr/bin/env python3
"""
scripts/preprocess.py

Downloads FLEURS (automatic) and preprocesses locally extracted Golos,
then caches everything to disk so training does not re-process on every run.

Usage:
    # FLEURS only (no Golos yet)
    python scripts/preprocess.py

    # FLEURS + Golos
    python scripts/preprocess.py --golos_path D:/golos

    # Golos only
    python scripts/preprocess.py --no_fleurs --golos_path D:/golos

    # Verify Golos directory layout without processing anything
    python scripts/preprocess.py --golos_path D:/golos --verify_only

    # Dry run: print dataset sizes, no preprocessing
    python scripts/preprocess.py --golos_path D:/golos --dry_run
"""

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess FLEURS + Golos for Whisper")
    p.add_argument("--config",        default="configs/acoustic.yaml")
    p.add_argument("--output_dir",    default="data/processed")
    p.add_argument("--golos_path",    default=None,
                   help="Root dir of extracted Golos dataset")
    p.add_argument("--golos_subsets", nargs="+", default=None,
                   metavar="SUBSET",
                   help="Which Golos sub-corpora to use: crowd farfield (default: both)")
    p.add_argument("--no_fleurs",     action="store_true", help="Skip FLEURS")
    p.add_argument("--verify_only",   action="store_true",
                   help="Check Golos directory layout and exit (no processing)")
    p.add_argument("--dry_run",       action="store_true",
                   help="Load raw datasets, print sizes, then exit")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # CLI overrides
    if args.no_fleurs:
        cfg["data"]["use_fleurs"] = False
    if args.golos_path:
        cfg["data"]["golos_path"] = args.golos_path
    if args.golos_subsets:
        cfg["data"]["golos_subsets"] = args.golos_subsets

    golos_path = cfg["data"].get("golos_path")

    # ---- Verify Golos layout ------------------------------------------------
    if args.verify_only:
        if not golos_path:
            logger.error("Pass --golos_path to verify Golos layout.")
            sys.exit(1)
        from src.acoustic.whisper.dataset import verify_golos_layout
        verify_golos_layout(golos_path)
        logger.info("Verification complete. No data was processed.")
        return

    # ---- Dry run ------------------------------------------------------------
    if args.dry_run:
        from datasets import load_dataset
        if cfg["data"].get("use_fleurs"):
            logger.info("--- FLEURS (google/fleurs, ru_ru) ---")
            ds = load_dataset("google/fleurs", "ru_ru", trust_remote_code=True)
            for split, data in ds.items():
                logger.info("  %-12s %6d samples", split, len(data))
        if golos_path:
            from src.acoustic.whisper.dataset import verify_golos_layout
            verify_golos_layout(golos_path)
        logger.info("Dry run complete.")
        return

    # ---- Validate we have at least one source --------------------------------
    if not cfg["data"].get("use_fleurs") and not golos_path:
        logger.error(
            "Nothing to process!\n"
            "Use --golos_path /path/to/golos and/or keep use_fleurs: true\n"
            "in configs/acoustic.yaml."
        )
        sys.exit(1)

    # ---- Load processor -----------------------------------------------------
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    from src.acoustic.whisper.dataset import build_dataset

    model_name = cfg["model"]["name"]
    logger.info("Loading processor from %s …", model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language=cfg["model"]["language"],
        task=cfg["model"]["task"],
    )

    # ---- Build + preprocess -------------------------------------------------
    logger.info("Building dataset for model WHISPER (this downloads data on first run) …")
    dataset = build_dataset(cfg, feature_extractor, tokenizer)

    # ---- Save to disk -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "acoustic_dataset")
    logger.info("Saving preprocessed dataset to %s …", save_path)
    dataset.save_to_disk(save_path)

    logger.info("Done! Split sizes:")
    for split, ds in dataset.items():
        logger.info("  %-12s %8d samples", split, len(ds))

    logger.info(
        "\nTo train:\n  python -m src.acoustic.whisper.train --config %s", args.config
    )


if __name__ == "__main__":
    main()
