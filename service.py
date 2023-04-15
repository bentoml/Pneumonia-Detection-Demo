from __future__ import annotations

import typing as t

import torch
import pydantic
import PIL.Image
import torchvision.transforms as T

import bentoml

pneumonia_model = bentoml.pytorch.get("resnet-pneumonia")
idx_to_class = t.cast(
    'dict[int, t.Literal["NORMAL", "PNEUMONIA"]]',
    pneumonia_model.info.metadata["idx2cls"],
)

pneumonia = pneumonia_model.to_runner()

svc = bentoml.Service("pneumonia-classifier", runners=[pneumonia])

preprocess = T.Compose(
    [
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class Output(pydantic.BaseModel):
    class_name: t.Literal["NORMAL", "PNEUMONIA"]

    @classmethod
    def from_result(cls, tensor: torch.Tensor) -> Output:
        _, pred = torch.max(tensor, 1)
        return cls(class_name=idx_to_class[pred.item()])


@svc.api(
    input=bentoml.io.Image(),
    output=bentoml.io.JSON.from_sample(sample=Output(class_name="PNEUMONIA")),
    route="/v1/classify",
)
async def classify(image: PIL.Image.Image) -> Output:
    input_tensor = t.cast("torch.Tensor", preprocess(image))
    res = await pneumonia.async_run(input_tensor.unsqueeze(0))
    return Output.from_result(res)
