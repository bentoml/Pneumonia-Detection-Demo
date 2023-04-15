from __future__ import annotations

import typing as t

import torch
import pydantic
import PIL.Image
import PIL.ImageOps

import bentoml

bento_vit_model = bentoml.transformers.get("vit-model-pneumonia")
model = bento_vit_model.to_runner()
extractor = bentoml.transformers.get("vit-extractor-pneumonia").to_runner()

svc = bentoml.Service("pneumonia-classifier", runners=[model, extractor])


def preprocess(image: PIL.Image.Image) -> PIL.Image.Image:
    return PIL.ImageOps.exif_transpose(image).convert("RGB")


class Output(pydantic.BaseModel):
    class_name: t.Literal["NORMAL", "PNEUMONIA"]

    @classmethod
    def from_result(cls, logits: torch.Tensor) -> Output:
        top_k = t.cast(int, bento_vit_model.info.metadata["top_k"])
        id2label = bento_vit_model.custom_objects["id2label"]

        # pneumonia model has two classes, which will return a list of two dicts.
        probs = logits.softmax(-1)[0]
        scores, ids = probs.topk(top_k)
        p0, p1 = [
            (score, id2label[id_]) for score, id_ in zip(scores.tolist(), ids.tolist())
        ]
        return cls(class_name=p0[1] if p0[0] > p1[0] else p1[1])


@svc.api(
    input=bentoml.io.Image(),
    output=bentoml.io.JSON(pydantic_model=Output),
    route="/v1/classify",
)
async def classify(image: PIL.Image.Image) -> Output:
    image = preprocess(image)
    features = await extractor.async_run(images=image, return_tensors="pt")
    outputs = await model.async_run(**features)
    return Output.from_result(outputs.logits)
