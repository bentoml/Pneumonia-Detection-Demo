from __future__ import annotations

import pytest
import typing as t
import contextlib
import bentoml
from bentoml._internal.utils import cached_contextmanager

if t.TYPE_CHECKING:
    from bentoml.server import Server


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "server" in metafunc.fixturenames:
        metafunc.parametrize("server", create_server())

# TODO: Add containerize tests
@cached_contextmanager("{project_path}, {cleanup}")
def build(
    project_path: str, cleanup: bool = True
) -> t.Generator[bentoml.Bento, None, None]:
    """
    Build a BentoML project.
    """
    from bentoml import bentos

    print(f"Building bento: {project_path}")
    bento = bentos.build_bentofile(build_ctx=project_path)
    yield bento
    if cleanup:
        print(f"Deleting bento: {str(bento.tag)}")
        bentos.delete(bento.tag)

def get_bento_name(project_path: str) -> str:
    from bentoml._internal.bento.build_config import BentoBuildConfig


def create_server() -> list[Server]:
    stack = contextlib.ExitStack()

    stack.close()
