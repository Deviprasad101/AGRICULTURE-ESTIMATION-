#!/usr/bin/env python3
"""
Field boundary detection pipeline template: Fields of The World (FTW) + GeoAI Mask R-CNN.

Focus: Andhra Pradesh / India — use FTW country="india" for training chips; clip predictions to AP
using your own AOI or district/mandal boundaries.

Requires: pip install geoai-py geopandas

Usage:
  python scripts/ap_field_boundaries_ftw_geoai.py --country india --download-only
  python scripts/ap_field_boundaries_ftw_geoai.py --country india --prepare-only
  python scripts/ap_field_boundaries_ftw_geoai.py --country luxembourg --train --epochs 5

Training/inference need a suitable PyTorch environment (GPU recommended).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _require_geoai():
    try:
        import geoai  # noqa: F401
    except ImportError as e:
        print("Missing dependency. Install with: pip install geoai-py geopandas", file=sys.stderr)
        raise SystemExit(1) from e
    return __import__("geoai")


def cmd_download(geoai, country: str, output_dir: str) -> None:
    geoai.download_ftw(countries=[country.lower()], output_dir=output_dir)
    print(f"Download complete: {os.path.join(output_dir, country.lower())}")


def cmd_prepare(geoai, ftw_root: str, country: str) -> dict:
    data = geoai.prepare_ftw(ftw_root, country=country.lower())
    print("Prepared layout:", {k: data[k] for k in data if isinstance(data[k], (str, Path))})
    return data


def cmd_train(geoai, data: dict, out_models: str, epochs: int, batch_size: int) -> None:
    geoai.train_instance_segmentation_model(
        images_dir=data["images_dir"],
        labels_dir=data["labels_dir"],
        output_dir=out_models,
        num_classes=2,
        num_channels=4,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=0.005,
        val_split=0.2,
        instance_labels=True,
        visualize=True,
        verbose=True,
    )
    print(f"Training finished. Models under {out_models}")


def cmd_infer_sample(geoai, data: dict, model_path: str, out_tif: str) -> None:
    test_dir = Path(data["test_dir"])
    test_images = sorted(test_dir.glob("*.tif"))
    if not test_images:
        print("No test GeoTIFFs found in", test_dir, file=sys.stderr)
        raise SystemExit(2)
    test_image_path = str(test_images[0])
    geoai.instance_segmentation(
        input_path=test_image_path,
        output_path=out_tif,
        model_path=model_path,
        num_classes=2,
        num_channels=4,
        window_size=256,
        overlap=128,
        confidence_threshold=0.5,
        batch_size=4,
        vectorize=True,
        class_names=["background", "field"],
    )
    print("Wrote", out_tif, "(and sidecar outputs from geoai)")


def main() -> int:
    p = argparse.ArgumentParser(description="FTW + GeoAI field boundaries (India / AP workflow)")
    p.add_argument("--country", default="india", help="FTW country code, e.g. india, luxembourg")
    p.add_argument("--ftw-root", default="ftw_data", help="Root folder for download_ftw / prepare_ftw")
    p.add_argument("--models-dir", default="field_boundaries/models", help="Model output directory")
    p.add_argument("--download-only", action="store_true", help="Only run geoai.download_ftw")
    p.add_argument("--prepare-only", action="store_true", help="Only run geoai.prepare_ftw (after download)")
    p.add_argument("--train", action="store_true", help="Train Mask R-CNN (requires prepare first)")
    p.add_argument("--infer-sample", action="store_true", help="Run inference on first test chip")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--model-path", default="field_boundaries/models/best_model.pth")
    p.add_argument("--prediction-tif", default="field_boundary_prediction.tif")
    args = p.parse_args()
    country = args.country.lower()
    any_op = args.download_only or args.prepare_only or args.train or args.infer_sample
    if not any_op:
        p.print_help()
        ex = sys.argv[0]
        print(
            "\nExample (India, AP-oriented):\n"
            f"  1) python {ex} --country india --download-only\n"
            f"  2) python {ex} --country india --prepare-only\n"
            f"  3) python {ex} --country india --train --epochs 20\n"
            f"  4) python {ex} --country india --infer-sample"
        )
        return 0

    geoai = _require_geoai()

    if args.download_only:
        cmd_download(geoai, country, args.ftw_root)
        return 0

    if args.prepare_only:
        cmd_prepare(geoai, args.ftw_root, country)
        return 0

    if args.train:
        data = cmd_prepare(geoai, args.ftw_root, country)
        cmd_train(geoai, data, args.models_dir, args.epochs, args.batch_size)
        return 0

    if args.infer_sample:
        data = cmd_prepare(geoai, args.ftw_root, country)
        mp = args.model_path
        if not os.path.isfile(mp):
            print("Model not found:", mp, file=sys.stderr)
            print("Train first with --train or pass --model-path", file=sys.stderr)
            return 2
        cmd_infer_sample(geoai, data, mp, args.prediction_tif)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
