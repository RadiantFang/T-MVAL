import logging
import warnings

import pytorch_lightning as pl
import torch
from enformer_pytorch import Enformer
from enformer_pytorch.data import str_to_one_hot
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings(
    "ignore",
    message=".*Consider installing `litmodels` package.*",
    category=UserWarning,
)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class EnformerModel(pl.LightningModule):
    """Single-task regression wrapper around Enformer."""

    def __init__(
        self,
        lr=1e-4,
        loss="mse",
        pretrained=True,
        dim=1536,
        depth=11,
        n_downsamples=7,
    ):
        super().__init__()
        self.n_tasks = 1
        self.save_hyperparameters()

        if pretrained:
            self.trunk = Enformer.from_pretrained(
                "/root/enformermodel", target_length=-1
            )._trunk
        else:
            self.trunk = Enformer.from_hparams(
                dim=dim,
                depth=depth,
                heads=8,
                num_downsamples=n_downsamples,
                target_length=-1,
            )._trunk
        self.head = nn.Linear(dim * 2, self.n_tasks, bias=True)

        self.lr = lr
        self.loss_type = loss
        if loss == "poisson":
            self.loss = nn.PoissonNLLLoss(log_input=True, full=True)
        elif loss == "mse":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss}")

        self.val_losses = []

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, strict=True, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("state_dict", checkpoint)
        hyper_parameters = checkpoint.get("hyper_parameters", {})

        init_kwargs = {
            "lr": hyper_parameters.get("lr", 1e-4),
            "loss": hyper_parameters.get("loss", "mse"),
            **kwargs,
        }
        init_kwargs["pretrained"] = False

        model = cls(**init_kwargs)
        model.load_state_dict(state_dict, strict=strict)
        return model

    def forward(self, x, return_logits=False):
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], str):
                x = str_to_one_hot(x)
            else:
                x = x[0]

        x = self.trunk(x)
        x = self.head(x)
        x = x.mean(1)
        x = x.squeeze(-1)

        if self.loss_type == "poisson" and not return_logits:
            x = torch.exp(x)
        return x

    def _compute_loss(self, batch):
        x, y = batch
        logits = self(x, return_logits=True)
        return self.loss(logits, y)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log(
            "train_loss", loss, logger=True, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.val_losses.append(loss.detach().cpu())
        self.log("val_loss", loss, logger=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_val_loss = torch.mean(torch.stack(self.val_losses)).item()
            print(f"\n[Validation] Mean validation loss: {avg_val_loss:.6f}")
            self.val_losses.clear()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    def train_on_dataset(
        self,
        train_dataset,
        val_dataset,
        device=0,
        batch_size=512,
        num_workers=1,
        save_dir=".",
        max_epochs=10,
        weights=None,
    ):
        torch.set_float32_matmul_precision("medium")

        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = [device]
        else:
            accelerator = "cpu"
            devices = 1

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=CSVLogger(save_dir),
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    filename="sknsh_best-model-{epoch:02d}-{val_loss:.4f}",
                    verbose=True,
                )
            ],
            enable_model_summary=False,
        )

        if weights is None:
            train_dl = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        else:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            train_dl = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=sampler,
            )

        val_dl = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        trainer.fit(model=self, train_dataloaders=train_dl, val_dataloaders=val_dl)
        return trainer

    def predict_on_dataset(
        self,
        dataset,
        device=0,
        num_workers=0,
        batch_size=512,
    ):
        torch.set_float32_matmul_precision("medium")

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if isinstance(device, int):
            prediction_device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            prediction_device = torch.device(device)

        was_training = self.training
        self.to(prediction_device)
        self.eval()

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                if torch.is_tensor(batch):
                    batch = batch.to(prediction_device)
                elif isinstance(batch, (list, tuple)):
                    batch = type(batch)(
                        item.to(prediction_device) if torch.is_tensor(item) else item
                        for item in batch
                    )
                predictions.append(self(batch))

        if was_training:
            self.train()

        return torch.cat(predictions).cpu().detach().numpy()
