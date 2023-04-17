from __future__ import annotations

import copy
import time
import typing as t
import pathlib as P
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as M
import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data import DataLoader

import bentoml

ROOT = P.Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    .decode("utf-8")
    .strip()
)
DATASET_DIR = ROOT.joinpath("data", "chest_xray", "chest_xray")
# number of iterations
EPOCHS = 20


def transform(phase: t.Literal["train", "test", "val"] = "train") -> T.Compose:
    if phase == "train":
        return T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def init_model(
    num_classes: int = 2, backbone: t.Literal["resnet", "vgg"] = "vgg"
) -> M.ResNet | M.VGG:
    def classifier(num_inputs):
        return nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1),
        )

    if backbone == "resnet":
        model = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
    elif backbone == "vgg":
        model = M.vgg16(weights=M.VGG16_Weights.DEFAULT)
    else:
        raise ValueError(
            f"Invalid backbone {backbone}, currently only supports resnet50 or vgg16"
        )

    for param in model.parameters():
        param.requires_grad = False

    if backbone == "resnet":
        n_inputs = model.fc.in_features
        model.fc = nn.Linear(n_inputs, num_classes)
    elif backbone == "vgg":
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = classifier(n_inputs)

    return model.to(torch.float32).to("cuda:0" if torch.cuda.is_available() else "cpu")


class_to_idx, idx_to_class = (
    {"NORMAL": 0, "PNEUMONIA": 1},
    {0: "NORMAL", 1: "PNEUMONIA"},
)


def prepare_dataloader(directory: str | None = None) -> tuple[DataLoader, ...]:
    datapath = DATASET_DIR
    if directory is not None:
        datapath = P.Path(directory)

    train_test_eval = tuple(
        D.ImageFolder(datapath.joinpath(phase).__fspath__(), transform=transform(phase))
        for phase in ("train", "test", "val")
    )

    return tuple(
        DataLoader(sets, batch_size=64, shuffle=True) for sets in train_test_eval
    )


def train_model(
    model, criterion, optimizer, train_loader, test_loader, scheduler, num_epochs=EPOCHS
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {
        "train": len(train_loader.dataset),
        "test": len(test_loader.dataset),
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in {"train": train_loader, "test": test_loader}[phase]:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            else:
                print(f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def accuracy(
    outputs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds


if __name__ == "__main__":
    backbone = "resnet"
    model = init_model(backbone=backbone)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_loader, test_loader, _ = prepare_dataloader()

    model = train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        scheduler,
    )

    bentomodel = bentoml.pytorch.save_model(
        f"{backbone}-pneumonia",
        model,
        metadata={
            "idx2cls": idx_to_class,
            "cls2idx": class_to_idx,
            "backbone": backbone,
        },
    )

    print("Saved model:", bentomodel)
