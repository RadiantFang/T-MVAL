import os
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Diffusion.diffusion import Diffusion
from Diffusion.unet import UNet

DATA_SPLITS_PATH = "./x_top_splits.pt"
CHECKPOINT_DIR = "./checkpoints"


def prepare_model_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)

    if x.ndim != 4:
        raise ValueError(f"Expected input with 4 dimensions, got shape {tuple(x.shape)}")

    return x


def get_autocast_device_type(device: Any) -> str:
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def train_step(
    x: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Any,
    precision: str | None = None,
):
    if precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    autocast_device_type = get_autocast_device_type(device)
    x = prepare_model_input(x)
    x = x.to(device, dtype=torch.float32)
    with torch.autocast(
        device_type=autocast_device_type,
        dtype=dtype,
        enabled=autocast_device_type != "cpu",
    ):
        loss = model(x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.mean().item()


@torch.no_grad()
def val_step(
    x: torch.Tensor,
    model: torch.nn.Module,
    device: Any,
    precision: str | None = None,
):
    if precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    autocast_device_type = get_autocast_device_type(device)
    x = prepare_model_input(x)
    x = x.to(device, dtype=torch.float32)
    with torch.autocast(
        device_type=autocast_device_type,
        dtype=dtype,
        enabled=autocast_device_type != "cpu",
    ):
        loss = model(x)
    return loss.mean().item()


@torch.no_grad()
def evaluate_loader(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: Any,
    precision: str | None = None,
) -> float:
    was_training = model.training
    model.eval()
    losses = [val_step(x, model, device, precision) for x in dataloader]
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else float("inf")


class SequenceDataset(Dataset):
    def __init__(self, seqs: torch.Tensor):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.seqs[index]


def build_dataloaders(batch_size, num_workers, pin_memory):
    split_dict = torch.load(DATA_SPLITS_PATH, map_location="cpu")
    x_train = split_dict["train"].float()
    x_val = split_dict["val"].float()
    x_test = split_dict["test"].float()

    train_dl = DataLoader(
        SequenceDataset(x_train),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_dl = DataLoader(
        SequenceDataset(x_val),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    test_dl = DataLoader(
        SequenceDataset(x_test),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_dl, val_dl, test_dl


def train(
    precision: str,
    num_workers: int,
    pin_memory: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_epochs: int,
    min_epochs: int,
    patience: int,
) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    local_batch_size = batch_size

    train_dl, val_dl, test_dl = build_dataloaders(local_batch_size, num_workers, pin_memory)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    global_step = 0
    model.train()

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_files = []
    best_model_state = None

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        model.train()
        train_losses = []
        for x in tqdm(train_dl, desc=f"Batch (Epoch {epoch + 1})", leave=False):
            loss = train_step(x, model, optimizer, device, precision)
            train_losses.append(loss)
            global_step += 1

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float("inf")
        avg_val_loss = evaluate_loader(val_dl, model, device, precision)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()

            checkpoint_dict = {
                "model": best_model_state,
                "optimizer": best_optimizer_state,
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": best_val_loss,
            }
            checkpoint_file = os.path.join(
                CHECKPOINT_DIR,
                f"model_epoch{epoch}_step{global_step}_valloss_{best_val_loss:2f}.pt",
            )
            torch.save(checkpoint_dict, checkpoint_file)
            checkpoint_files.append(checkpoint_file)
            if len(checkpoint_files) > 2:
                os.remove(checkpoint_files.pop(0))
        else:
            patience_counter += 1

        if epoch >= min_epochs and patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}, Best val loss: {best_val_loss} "
                f"achieved at epoch {epoch - patience_counter}"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss = evaluate_loader(test_dl, model, device, precision)
    print(f"Final test loss using best validation checkpoint: {test_loss:.6f}")


def main() -> None:
    model = UNet(dim=200, channels=1, dim_mults=[1, 2, 4], resnet_block_groups=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    diffusion = Diffusion(model, timesteps=50, beta_start=0.0001, beta_end=0.2)
    train(
        model=diffusion,
        optimizer=optimizer,
        precision="bf16",
        num_workers=2,
        pin_memory=True,
        batch_size=256,
        num_epochs=200,
        min_epochs=100,
        patience=10,
    )


if __name__ == "__main__":
    main()
