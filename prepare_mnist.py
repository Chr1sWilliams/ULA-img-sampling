#!/usr/bin/env python3
"""Download a small MNIST subset and export it as training PNG files."""

import argparse
import gzip
from pathlib import Path
import struct
from urllib.request import urlretrieve

from PIL import Image


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
IMAGE_ARCHIVE = "train-images-idx3-ubyte.gz"
LABEL_ARCHIVE = "train-labels-idx1-ubyte.gz"


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=2048)
    return parser


def _download(download_dir: Path, filename: str) -> Path:
    destination = download_dir / filename
    if not destination.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {filename}...")
        urlretrieve(f"{MNIST_BASE_URL}/{filename}", destination)
    return destination


def _read_mnist(
    image_archive: Path,
    label_archive: Path,
) -> tuple[bytes, bytes, int, int, int]:
    with gzip.open(image_archive, "rb") as image_file:
        image_payload = image_file.read()
    with gzip.open(label_archive, "rb") as label_file:
        label_payload = label_file.read()

    image_magic, image_count, rows, columns = struct.unpack(
        ">IIII",
        image_payload[:16],
    )
    label_magic, label_count = struct.unpack(">II", label_payload[:8])
    if image_magic != 2051 or label_magic != 2049:
        raise ValueError("Downloaded MNIST files have invalid IDX headers.")
    if image_count != label_count:
        raise ValueError("MNIST image and label counts do not match.")
    return image_payload[16:], label_payload[8:], image_count, rows, columns


def main() -> None:
    args = create_argparser().parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive.")

    download_dir = Path(args.download_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_images = list(output_dir.glob("*.png"))
    if len(existing_images) >= args.limit:
        print(
            f"MNIST training images ready: {args.limit} files in {output_dir}"
        )
        return

    image_archive = _download(download_dir, IMAGE_ARCHIVE)
    label_archive = _download(download_dir, LABEL_ARCHIVE)
    images, labels, dataset_size, rows, columns = _read_mnist(
        image_archive,
        label_archive,
    )

    image_count = min(args.limit, dataset_size)
    pixels_per_image = rows * columns
    for index in range(image_count):
        label = labels[index]
        output_path = output_dir / f"{label}_{index:05d}.png"
        if not output_path.exists():
            start = index * pixels_per_image
            image = Image.frombytes(
                "L",
                (columns, rows),
                images[start : start + pixels_per_image],
            )
            image.save(output_path)

    print(f"MNIST training images ready: {image_count} files in {output_dir}")


if __name__ == "__main__":
    main()
