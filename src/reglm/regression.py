import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from enformer_pytorch import Enformer
from enformer_pytorch.data import str_to_one_hot
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class SeqDataset(Dataset):
    """
    PyTorch dataset class for training enformer-based regression models

    Args:
        seqs (list, pd.DataFrame): either a list of DNA sequences, or a dataframe whose
            first column is DNA sequences and remaining columns are labels.
        seq_len (int): Length of sequences to return. Sequences will be padded with Ns
            on the right to reach this length.
    """

    def __init__(self, seqs, seq_len=None):
        super().__init__()

        self.is_labeled = False

        # Add seqs
        if isinstance(seqs, pd.DataFrame):
            self.seqs = seqs.iloc[:, 0]
            self.labels = torch.Tensor(
                seqs.values[:, 1:].astype(np.float32)
            )  # N, n_tasks
            self.is_labeled = True
        else:
            self.seqs = seqs

        # Add max sequence length
        self.seq_len = seq_len or np.max([len(seq) for seq in self.seqs])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Get sequence
        seq = self.seqs[idx]

        # Get label and augment sequence
        if self.is_labeled:
            label = self.labels[idx]

        # Pad sequence
        if len(seq) < self.seq_len:
            seq = seq + "N" * (self.seq_len - len(seq))

        # One-hot encode
        seq = str_to_one_hot(seq)

        if self.is_labeled:
            return seq, label
        else:
            return seq


class EnformerModel(pl.LightningModule):
    """
    Enformer-based single-task regression models that can be
    trained from scratch or finetuned.

    Args:
        lr (float): learning rate
        loss (str): "poisson" or "mse"
        pretrained (bool): If true, initialize from the pretrained enformer model
        dim (int): Number of conv layer filters
        depth (int): Number of transformer layers
        n_downsamples (int): Number of conv/pool blocks
    """

    def __init__(
        self,
        lr=1e-4,
        loss="poisson",
        pretrained=False,
        dim=1536,
        depth=11,
        n_downsamples=7,
    ):
        super().__init__()
        self.n_tasks = 1
        self.save_hyperparameters(ignore=["model"])

        # Build model
        if pretrained:
            self.trunk = Enformer.from_pretrained(
                "EleutherAI/enformer-official-rough", target_length=-1
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

        # Training params
        self.lr = lr
        self.loss_type = loss
        if loss == "poisson":
            self.loss = nn.PoissonNLLLoss(log_input=True, full=True)
        else:
            self.loss = nn.MSELoss()

    def forward(self, x, return_logits=False):
        if (isinstance(x, list)) or (isinstance(x, tuple)):
            if isinstance(x[0], str):
                # If x is a list of strings, convert it into a one-hot encoded tensor
                x = str_to_one_hot(x)
            else:
                # If x is a list (input, target) use only the input
                x = x[0]

        x = self.trunk(x)  # N, L, dim*2
        x = self.head(x)  # N, L, 1
        x = x.mean(1)  # N, 1

        if (self.loss_type == "poisson") and (not return_logits):
            x = torch.exp(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x, return_logits=True)
        loss = self.loss(logits, y)
        self.log(
            "train_loss", loss, logger=True, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x, return_logits=True)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, output):
        print("\nval_loss", torch.mean(torch.Tensor(output)).item())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

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

        # Make trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=[device],
            logger=CSVLogger(save_dir),
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min")],
        )
        # Make dataloaders
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

        # Fit model
        trainer.fit(model=self, train_dataloaders=train_dl, val_dataloaders=val_dl)
        return trainer

    def predict_on_dataset(
        self,
        dataset,
        device=0,
        num_workers=1,
        batch_size=512,
    ):
        torch.set_float32_matmul_precision("medium")

        # Make dataloader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Make trainer
        trainer = pl.Trainer(accelerator="gpu", devices=[device], logger=None)

        # Run inference
        return (
            torch.concat(trainer.predict(self, dataloader)).cpu().detach().numpy()
        )  # N, 1


class MultiTaskEnformerModel(nn.Module):
    """
    Combine multiple single-task enformer models into a single object.

    Args:
        models (list): List of multiple EnformerModel objects
        device (int): GPU index
    """

    def __init__(self, model1, model2, model3=None, mean=False, specificity=None):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.n_tasks = 2 if model3 is None else 3
        self.mean = mean
        self.specificity = specificity

    def forward(self, x):
        if self.model3 is None:
            x = [self.model1(x), self.model2(x)]
        else:
            x = [self.model1(x), self.model2(x), self.model3(x)]
        x = torch.cat(x, axis=1)  # N, n_models
        return x

    def predict_on_dataset(self, ds, **kwargs):
        if self.model3 is None:
            out = np.concatenate(
                [
                    self.model1.predict_on_dataset(ds, **kwargs),
                    self.model2.predict_on_dataset(ds, **kwargs),
                ],
                axis=1,
            )
        else:
            out = np.concatenate(
                [
                    self.model1.predict_on_dataset(ds, **kwargs),
                    self.model2.predict_on_dataset(ds, **kwargs),
                    self.model3.predict_on_dataset(ds, **kwargs),
                ],
                axis=1,
            )
        if self.mean:
            out = out.mean(1)
        elif self.specificity is not None:
            non_specific = [x for x in range(self.n_tasks) if x != self.specificity]
            out = out[:, self.specificity] - np.max(out[:, non_specific], axis=1)
        return out
