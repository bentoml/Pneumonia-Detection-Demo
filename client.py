from __future__ import annotations

from pathlib import Path

import PIL.Image

import bentoml

PROJECT_PATH = Path(__file__).parent

SAMPLES = PROJECT_PATH.joinpath("samples").glob("*")


def call(host: str = "127.0.0.1") -> None:
    with bentoml.SyncHTTPClient(f"http://{host}:3000") as client:
        for sample in SAMPLES:
            im = PIL.Image.open(sample)
            print("Prediction:", client.classify(image=im))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()

    call(host=args.host)
