from __future__ import annotations

import sys
import subprocess
import os
import io
import pytest
from pathlib import Path
import PIL.Image
import typing as t
import bentoml

if t.TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest

    P = t.TypeVar("P")
    Generator = t.Generator[P, None, None]

PROJECT_PATH = Path(__file__).parent.parent
BENTO_NAME = "pneumonia-classifier"


@pytest.fixture(
    name="xray_image",
    params=tuple(i.__fspath__() for i in PROJECT_PATH.joinpath("samples").glob("*")),
    scope="session",
)
def fixture_xray_im(request: FixtureRequest):
    with open(request.param, "rb") as f:
        im = PIL.Image.open(io.BytesIO(f.read()))
    return im


@pytest.fixture(name="project_path", params=[PROJECT_PATH], scope="session")
def fixture_project_path(request: FixtureRequest):
    return request.param


@pytest.fixture(autouse=True, scope="session")
def bento_directory(request: FixtureRequest):
    os.chdir(PROJECT_PATH.__fspath__())
    sys.path.insert(0, PROJECT_PATH.__fspath__())
    yield
    os.chdir(request.config.invocation_dir)
    sys.path.pop(0)


# TODO: Add containerize tests
@pytest.fixture(name="bento", scope="function")
def fixture_build_bento() -> Generator[bentoml.Bento]:
    try:
        bento = bentoml.get(BENTO_NAME)
    except bentoml.exceptions.NotFound:
        print(f"Building bento from path: {PROJECT_PATH}")
        subprocess.check_output(["bentoml", "build", "."])
        bento = bentoml.get(BENTO_NAME)
    yield bento
