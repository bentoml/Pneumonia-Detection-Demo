from __future__ import annotations

import subprocess
import os
import io
import pytest
from pathlib import Path
import PIL.Image
import typing as t
import contextlib
import bentoml
from bentoml._internal.utils import reserve_free_port

if t.TYPE_CHECKING:
    from _pytest.python import Metafunc
    from bentoml.server import Server
    from bentoml.client import Client
    from _pytest.fixtures import FixtureRequest

    P = t.TypeVar("P")
    Generator = t.Generator[P, None, None]

PROJECT_PATH = Path(__file__).parent.parent
BENTO_NAME = "pneumonia-classifier"


def pytest_generate_tests(metafunc: Metafunc) -> None:
    def parametrize(**kwargs):
        with create_server(**kwargs) as (server, client):
            if "server" in metafunc.fixturenames and "client" in metafunc.fixturenames:
                metafunc.parametrize("server,client", [(server, client)])
            elif "client" in metafunc.fixturenames:
                metafunc.parametrize("client", [client])

    function_name = metafunc.function.__qualname__
    if function_name.endswith("http"):
        parametrize(server_type="http")
    elif function_name.endswith("grpc"):
        parametrize(server_type="grpc")


@pytest.fixture(
    name="xray_image",
    params=tuple(i.__fspath__() for i in PROJECT_PATH.joinpath("samples").glob("*")),
    scope="session",
)
def fixture_xray_im(request: FixtureRequest):
    with open(request.param, "rb") as f:
        im = PIL.Image.open(io.BytesIO(f.read()))
    return im


@pytest.fixture(autouse=True, scope="package")
def bento_directory(request: FixtureRequest):
    os.chdir(PROJECT_PATH.__fspath__())
    yield
    os.chdir(request.config.invocation_dir)


# TODO: Add containerize tests
@contextlib.contextmanager
def build(project_path: str, cleanup: bool = True) -> Generator[bentoml.Bento]:
    """
    Build a BentoML project.
    """
    print(f"Building bento from path: {project_path}")
    subprocess.check_output(["bentoml", "build", project_path])
    bento = bentoml.get(BENTO_NAME)
    yield bento
    if cleanup:
        print(f"Deleting bento: {str(bento.tag)}")
        bentoml.bentos.delete(bento.tag)


@contextlib.contextmanager
def create_server(
    server_type: t.Literal["grpc", "http"] = "http",
    host: str = "127.0.0.1",
    cleanup: bool = True,
) -> Generator[tuple[Server, Client]]:
    stack = contextlib.ExitStack()
    server_cls = {"grpc": "GrpcServer", "http": "HTTPServer"}

    with reserve_free_port(
        host=host, enable_so_reuseport=server_type == "grpc"
    ) as server_port:
        pass
    copied = os.environ.copy()
    copied["BENTOML_CONFIG"] = PROJECT_PATH.joinpath(
        "config", "default.yaml"
    ).__fspath__()

    try:
        bento = bentoml.get(BENTO_NAME)
    except bentoml.exceptions.NotFound:
        # XXX: If passing abspath, this will fail
        bento = stack.enter_context(build(".", cleanup=False))

    kwargs = {"bento": bento, "production": True, "host": host, "port": server_port}

    try:
        server = getattr(bentoml, server_cls[server_type])(**kwargs)
        # client = stack.enter_context(server.start(env=copied))
        client = stack.enter_context(server.start())
        yield server, client
    finally:
        if cleanup:
            stack.close()
