from __future__ import annotations

import time
import pytest
import typing as t
import pydantic
from bentoml.testing.utils import async_request
from bentoml._internal.utils import LazyLoader

if t.TYPE_CHECKING:
    import PIL.Image
    from bentoml.server import Server
    from bentoml.client import Client, HTTPClient, GrpcClient
    from grpc_health.v1 import health_pb2 as pb_health
    from google.protobuf import struct_pb2 as pb_struct
else:
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")
    pb_struct = LazyLoader("pb_struct", globals(), "google.protobuf.struct_pb2")


@pytest.mark.asyncio
async def test_api_server_meta_http(client: Client) -> None:
    status, _, _ = await async_request("GET", f"http://{client.server_url}/")
    assert status == 200
    status, _, _ = await async_request("GET", f"http://{client.server_url}/healthz")
    assert status == 200
    status, _, _ = await async_request("GET", f"http://{client.server_url}/livez")
    assert status == 200
    status, _, _ = await async_request("GET", f"http://{client.server_url}/docs.json")
    assert status == 200
    status, _, body = await async_request("GET", f"http://{client.server_url}/metrics")
    assert status == 200
    assert body


@pytest.mark.asyncio
async def test_inference_http(
    server: Server, client: HTTPClient, xray_image: PIL.Image.Image
):
    assert client.health().status == 200

    res = await client.async_call("classify", inp=xray_image)
    # FIXME: Currently the models will always converge to PNEUMONIA even if the inputs is a NORMAL case.
    assert isinstance(res, pydantic.BaseModel)

    server.stop()

    timeout = 10
    start_time = time.time()
    while time.time() - start_time < timeout:
        retcode = server.process.poll()
        if retcode is not None and retcode <= 0:
            break


@pytest.mark.asyncio
async def test_inference_grpc(
    server: Server, client: GrpcClient, xray_image: PIL.Image.Image
):
    res = await client.health("bentoml.grpc.v1.BentoService")
    assert res.status == pb_health.HealthCheckResponse.SERVING

    res = await client.async_call("classify", xray_image)
    # FIXME: Currently the models will always converge to PNEUMONIA even if the inputs is a NORMAL case.
    assert res.json and isinstance(res.json, pb_struct.Value)

    server.stop()

    timeout = 10
    start_time = time.time()
    while time.time() - start_time < timeout:
        retcode = server.process.poll()
        if retcode is not None and retcode <= 0:
            break
