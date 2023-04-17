from __future__ import annotations

import os
import contextlib
from pathlib import Path
import pytest
import typing as t
import bentoml
from bentoml.testing.utils import async_request
from bentoml._internal.utils import LazyLoader, reserve_free_port

if t.TYPE_CHECKING:
    import PIL.Image
    from bentoml.client import HTTPClient, GrpcClient, Client
    from grpc_health.v1 import health_pb2 as pb_health
    from google.protobuf import struct_pb2 as pb_struct

    P = t.TypeVar("P")
    Generator = t.Generator[P, None, None]
else:
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")
    pb_struct = LazyLoader("pb_struct", globals(), "google.protobuf.struct_pb2")


@contextlib.contextmanager
def create_server(
    bento: bentoml.Bento,
    project_path: Path,
    host: str = "127.0.0.1",
    server_type: t.Literal["grpc", "http"] = "http",
) -> Generator[Client]:
    with reserve_free_port(
        host=host, enable_so_reuseport=server_type == "grpc"
    ) as server_port:
        pass
    copied = os.environ.copy()
    copied["BENTOML_CONFIG"] = project_path.joinpath(
        "config", "default.yaml"
    ).__fspath__()

    kwargs: dict[str, t.Any] = {
        "bento": bento,
        "production": True,
        "host": host,
        "port": server_port,
    }

    if server_type == "http":
        with bentoml.HTTPServer(**kwargs).start() as client:
            yield client
    elif server_type == "grpc":
        with bentoml.GrpcServer(**kwargs).start() as client:
            yield client
    else:
        raise ValueError(f"Unknown server type: {server_type}")


@pytest.mark.asyncio
async def test_api_server_meta_http(bento: bentoml.Bento, project_path: Path) -> None:
    with create_server(bento, project_path) as client:
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
    bento: bentoml.Bento, xray_image: PIL.Image.Image, project_path: Path
):
    with create_server(bento, project_path) as client:
        client = t.cast("HTTPClient", client)
        resp = await client.async_health()
        assert resp.status == 200

        res = await client.async_classify(xray_image)
        # FIXME: Currently the models will always converge to PNEUMONIA even if the inputs is a NORMAL case.
        assert isinstance(res, dict)


@pytest.mark.asyncio
async def test_inference_grpc(
    bento: bentoml.Bento, xray_image: PIL.Image.Image, project_path: Path
):
    with create_server(
        bento, project_path, host="0.0.0.0", server_type="grpc"
    ) as client:
        client = t.cast("GrpcClient", client)
        res = await client.health("bentoml.grpc.v1.BentoService")
        assert res.status == pb_health.HealthCheckResponse.SERVING

        res = await client.async_call("classify", xray_image)
        # FIXME: Currently the models will always converge to PNEUMONIA even if the inputs is a NORMAL case.
        assert res.json and isinstance(res.json, pb_struct.Value)
