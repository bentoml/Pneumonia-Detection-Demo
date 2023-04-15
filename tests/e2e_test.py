from __future__ import annotations

import os
import asyncio
import platform
from pathlib import Path
import pytest
import typing as t
import bentoml
from bentoml.testing.utils import async_request
from bentoml._internal.utils import LazyLoader, reserve_free_port

if t.TYPE_CHECKING:
    import PIL.Image
    from bentoml.server import Server, HTTPServer, GrpcServer
    from bentoml.client import HTTPClient, GrpcClient
    from grpc_health.v1 import health_pb2 as pb_health
    from google.protobuf import struct_pb2 as pb_struct

    P = t.TypeVar("P")
    Generator = t.Generator[P, None, None]
else:
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")
    pb_struct = LazyLoader("pb_struct", globals(), "google.protobuf.struct_pb2")


@t.overload
def create_server(
    bento: bentoml.Bento, host: str = ..., server_type: t.Literal["grpc"] = ...
) -> GrpcServer:
    ...


@t.overload
def create_server(
    bento: bentoml.Bento, host: str = ..., server_type: t.Literal["http"] = ...
) -> HTTPServer:
    ...


@pytest.fixture(name="envvars", scope="function")
def fixture_envvars(project_path: Path) -> dict[str, t.Any]:
    copied = os.environ.copy()
    copied["BENTOML_CONFIG"] = project_path.joinpath(
        "config", "default.yaml"
    ).__fspath__()
    return copied


def create_server(
    bento: bentoml.Bento,
    host: str = "127.0.0.1",
    server_type: t.Literal["grpc", "http"] = "http",
) -> Server:
    with reserve_free_port(
        host=host, enable_so_reuseport=server_type == "grpc"
    ) as server_port:
        pass

    kwargs: dict[str, t.Any] = {
        "bento": bento,
        "production": True,
        "host": host,
        "port": server_port,
    }

    if server_type == "http":
        server = bentoml.HTTPServer(**kwargs)
    elif server_type == "grpc":
        server = bentoml.GrpcServer(**kwargs)
    else:
        raise ValueError(f"Unknown server type: {server_type}")
    server.timeout = 90
    return server


@pytest.mark.asyncio
async def test_api_server_meta_http(
    bento: bentoml.Bento, envvars: dict[str, t.Any]
) -> None:
    server = create_server(bento)

    with server.start(env=envvars) as client:
        status, _, _ = await async_request("GET", client.server_url)
        assert status == 200
        status, _, _ = await async_request("GET", f"{client.server_url}/healthz")
        assert status == 200
        status, _, _ = await async_request("GET", f"{client.server_url}/livez")
        assert status == 200
        status, _, _ = await async_request("GET", f"{client.server_url}/docs.json")
        assert status == 200
        status, _, body = await async_request("GET", f"{client.server_url}/metrics")
        assert status == 200
        assert body


@pytest.mark.asyncio
async def test_inference_http(
    bento: bentoml.Bento,
    xray_image: tuple[PIL.Image.Image, str],
    envvars: dict[str, t.Any],
):
    im, label = xray_image
    server = create_server(bento)

    with server.start(env=envvars) as client:
        client = t.cast("HTTPClient", client)
        resp = await client.async_health()
        assert resp.status == 200

        res = await client.async_classify(im)
        assert isinstance(res, dict) and res["class_name"] == label


@pytest.mark.skipif(
    platform.system() == "Windows", reason="gRPC is not yet fully supported on Windows"
)
@pytest.mark.asyncio
async def test_inference_grpc(
    bento: bentoml.Bento,
    xray_image: tuple[PIL.Image.Image, str],
    envvars: dict[str, t.Any],
):
    im, label = xray_image
    server = create_server(bento, host="0.0.0.0", server_type="grpc")

    with server.start(env=envvars) as client:
        client = t.cast("GrpcClient", client)
        # NOTE: we need to sleep for a bit to make sure the server is ready.
        await asyncio.sleep(20)
        res = await client.health("bentoml.grpc.v1.BentoService")
        assert res.status == pb_health.HealthCheckResponse.SERVING

        res = await client.async_call("classify", im)
        assert res.json and isinstance(res.json, pb_struct.Value)
        assert res.json.struct_value.fields["class_name"].string_value == label
