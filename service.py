from __future__ import annotations

import typing as t

import torch
import pydantic
import PIL.Image
import PIL.ImageOps

import bentoml

from save_model import download_model

_ = download_model()


def preprocess(image: PIL.Image.Image) -> PIL.Image.Image:
    return PIL.ImageOps.exif_transpose(image).convert("RGB")


class Output(pydantic.BaseModel):
    class_name: t.Literal["NORMAL", "PNEUMONIA"]


@bentoml.service(name="pneumonia-classifier")
class PneumoniaClassifier:
    vit_model = bentoml.models.BentoModel("vit-model-pneumonia")
    vit_extractor = bentoml.models.BentoModel("vit-extractor-pneumonia")

    def __init__(self) -> None:
        self.model = bentoml.transformers.load_model(self.vit_model)
        self.extractor = bentoml.transformers.load_model(self.vit_extractor)
        self.top_k = t.cast(int, self.vit_model.info.metadata["top_k"])
        self.id2label = self.vit_model.custom_objects["id2label"]

    @bentoml.api(route="/v1/classify")
    def classify(self, image: PIL.Image.Image) -> Output:
        features = self.extractor(images=preprocess(image), return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**features).logits

        # pneumonia model has two classes, which will return a list of two dicts.
        probs = logits.softmax(-1)[0]
        scores, ids = probs.topk(self.top_k)
        p0, p1 = [
            (score, self.id2label[id_])
            for score, id_ in zip(scores.tolist(), ids.tolist())
        ]
        return Output(class_name=p0[1] if p0[0] > p1[0] else p1[1])
