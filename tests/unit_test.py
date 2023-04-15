from __future__ import annotations


import asyncio
import torch
import pytest
import typing as t
from transformers.modeling_outputs import ImageClassifierOutput

if t.TYPE_CHECKING:
    import PIL.Image
    from pytest_mock import MockerFixture


def test_preprocess(xray_image: tuple[PIL.Image.Image, str]):
    import service

    im, _ = xray_image

    res = service.preprocess(im)
    assert res.mode == "RGB"


@pytest.mark.parametrize(
    "result_tensor,expected",
    [
        (torch.tensor([[-1.2620, 1.3168]]), "PNEUMONIA"),
        (torch.tensor([[1.4293, -1.5865]]), "NORMAL"),
    ],
)
def test_output_from_result(result_tensor: torch.Tensor, expected: int):
    import service

    assert expected == service.Output.from_result(result_tensor).class_name


@pytest.mark.asyncio
async def test_classify(mocker: MockerFixture, xray_image: tuple[PIL.Image.Image, str]):
    import service

    im, _ = xray_image

    e_runner = mocker.patch("service.extractor")
    e_runner.async_run = e_runner.object(service.extractor, "async_run")
    future = asyncio.Future()
    future.set_result({"pixel_values": torch.Tensor([1.0])})
    e_runner.async_run.return_value = future

    m_runner = mocker.patch("service.model")
    m_runner.async_run = m_runner.object(service.model, "async_run")
    future = asyncio.Future()
    future.set_result(
        ImageClassifierOutput(
            logits=torch.tensor([[-1.2620, 1.3168]], dtype=torch.float32)
        )
    )
    m_runner.async_run.return_value = future

    res = await service.classify(im)
    assert res.class_name == "PNEUMONIA"
