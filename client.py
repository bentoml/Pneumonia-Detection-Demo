from __future__ import annotations

from pathlib import Path

import PIL.Image

import bentoml

PROJECT_PATH = Path(__file__).parent

SAMPLES = PROJECT_PATH.joinpath("samples").glob("*")


def call(host: str = "127.0.0.1") -> None:
    client = bentoml.client.Client.from_url(f"{host}:3000")

    for sample in SAMPLES:
        im = PIL.Image.open(sample)
        print("Prediction:", client.classify(im))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()

    call(host=args.host)
