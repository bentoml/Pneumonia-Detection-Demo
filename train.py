from __future__ import annotations

import typing as t
import pathlib as P
import subprocess
from timeit import default_timer

import numpy as np
import torch
import pandas as pd
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
NUM_ITER = 10


def transform(phase: t.Literal["train", "test", "val"] = "train") -> T.Compose:
    if phase == "train":
        return T.Compose(
            [
                T.Resize(size=(256, 256)),
                T.RandomRotation(degrees=15),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def init_model(num_classes: int = 2) -> M.ResNet:
    model = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_inputs = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore
        nn.Linear(num_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1),
    )
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


def training(
    model: M.ResNet,
    criterion: nn.NLLLoss,
    optimizer: optim.Adam,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_file_name: str = "resnet50-pneumonia.pt",
    max_epochs_stop: int = 5,
    n_epochs: int = NUM_ITER,
    print_every: int = 2,
):
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    history = []
    best_epoch = 0
    overall_start = default_timer()

    # Main loop
    for epoch in range(n_epochs):
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = default_timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))

            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f"Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {default_timer() - start:.2f} seconds elapsed in epoch.",
                end="\r",
            )

        # After training loops ends, start validation
        else:
            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in val_loader:
                    # Tensors to gpu
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(val_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(val_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f"\nEpoch: {epoch} \ttraining loss: {train_loss:.4f} \ttraining accuracy: {100 * train_acc:.2f}%"
                    )
                    print(
                        f"\t\tvalid loss: {valid_loss:.4f}\tvalidation accuracy: {100 * valid_acc:.2f}%"
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f"\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%"
                        )
                        total_time = default_timer() - overall_start
                        print(
                            f"{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch."
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                "train_loss",
                                "valid_loss",
                                "train_acc",
                                "valid_acc",
                            ],
                        )
                        return model, history

    # Record overall time and print out stats
    total_time = default_timer() - overall_start
    print(
        f"\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%"
    )
    print(
        f"{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch."
    )
    # Format history
    history = pd.DataFrame(
        history, columns=["train_loss", "valid_loss", "train_acc", "valid_acc"]
    )
    return model, history


def accuracy(
    outputs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds


if __name__ == "__main__":
    model = init_model()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    train_loader, test_loader, val_loader = prepare_dataloader()

    model, _ = training(model, criterion, optimizer, train_loader, val_loader)

    bentomodel = bentoml.pytorch.save_model(
        "resnet-pneumonia",
        model,
        metadata={"idx2cls": idx_to_class, "cls2idx": class_to_idx},
    )

    print("Saved model:", bentomodel)
