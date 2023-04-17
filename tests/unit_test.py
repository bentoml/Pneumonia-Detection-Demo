from __future__ import annotations

import torch

import asyncio
import pytest
import typing as t

if t.TYPE_CHECKING:
    import PIL.Image
    from pytest_mock import MockerFixture


def test_preprocess(xray_image: PIL.Image.Image):
    import service

    res = t.cast("torch.Tensor", service.preprocess(xray_image))
    assert tuple(res.shape[:2]) == (3, 256)


@pytest.mark.parametrize(
    "result_tensor,expected",
    [(torch.tensor([[-2, 1.2]]), "PNEUMONIA"), (torch.tensor([[0, 0]]), "NORMAL")],
)
def test_output_from_result(result_tensor: torch.Tensor, expected: int):
    import service

    assert expected == service.Output.from_result(result_tensor).class_name


@pytest.mark.asyncio
async def test_classify(mocker: MockerFixture, xray_image: PIL.Image.Image):
    import service

    m_runner = mocker.patch("service.pneumonia")
    m_runner.async_run = m_runner.object(service.pneumonia, "async_run")
    future = asyncio.Future()
    future.set_result(torch.tensor([[-2, 1.2]]))
    m_runner.async_run.return_value = future

    res = await service.classify(xray_image)
    assert res.class_name == "PNEUMONIA"
