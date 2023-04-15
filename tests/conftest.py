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
    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config

    P = t.TypeVar("P")
    Generator = t.Generator[P, None, None]

PROJECT_PATH = Path(__file__).parent.parent
BENTO_NAME = "pneumonia-classifier"


def pytest_collection_modifyitems(
    session: Session, config: Config, items: list[Item]
) -> None:
    subprocess.check_call(
        [sys.executable, f"{os.path.join(PROJECT_PATH, 'save_model.py')}"]
    )


@pytest.fixture(
    name="xray_image",
    params=tuple(i.__fspath__() for i in PROJECT_PATH.joinpath("samples").glob("*")),
    scope="session",
)
def fixture_xray_im(request: FixtureRequest):
    with open(request.param, "rb") as f:
        im = PIL.Image.open(io.BytesIO(f.read()))
    # NOTE: The image from xray datasets labels NORMAL image within the filename.
    return im, "NORMAL" if "NORMAL" in request.param else "PNEUMONIA"


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
    bentoml.delete(BENTO_NAME)
