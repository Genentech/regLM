import numpy as np
import pandas as pd
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from datetime import datetime
sys.path.append('/code/hyena-dna')
sys.path.append("/code/hyena-reggpt/reglm/")
from src.models.sequence.long_conv_lm import ConvLMHeadModel


def load_pretrained_model(ckpt_dir='./checkpoints/', model='hyenadna-tiny-1k-seqlen'):

    # Check model name
    assert model in [
        "hyenadna-tiny-16k-seqlen-d128",
        "hyenadna-large-1m-seqlen",
        "hyenadna-medium-160k-seqlen",
        "hyenadna-medium-450k-seqlen",
        "hyenadna-small-32k-seqlen",
        "hyenadna-tiny-1k-seqlen",
        "hyenadna-tiny-1k-seqlen-d256",
    ]

    # Make directory if needed
    if not os.path.exists(ckpt_dir):
        print("Making checkpoint directory")
        os.makedirs(ckpt_dir)

    # Download model if not already downloaded
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print("Downloading model")
        os.system(f'wget -P {ckpt_dir} https://huggingface.co/LongSafari/{model}/resolve/main/config.json')
        os.system(f'wget -P {ckpt_dir} https://huggingface.co/LongSafari/{model}/resolve/main/weights.ckpt')

    # Load config
    config = json.load(open(os.path.join(ckpt_dir, 'config.json'), 'r'))

    # Generate model
    model = ConvLMHeadModel(**config)

    # Load weights
    state_dict = torch.load(os.path.join(ckpt_dir, 'weights.ckpt'), map_location=torch.device("cpu"))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict["state_dict"], "model."
    )
    model_state_dict = state_dict["state_dict"]
    for key in list(model_state_dict.keys()):
        if "torchmetrics" in key:
            model_state_dict.pop(key)
    
    model.load_state_dict(model_state_dict)
    return model


class LightningModel(pl.LightningModule):
    def __init__(self, config=None, ckpt_dir=None, logger=None, save_dir=".", lr=1e-4):
        super().__init__()
        self.save_dir = save_dir
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr

        # Build model
        if ckpt_dir is not None:
            self.model = load_pretrained_model(ckpt_dir)
        elif config is not None:
            self.model = ConvLMHeadModel(**config)
        else:
            raise ValueError('either config or ckpt_dir must be provided.')

        # Logger
        self.logger_type = logger
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=-1)
        self.val_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=-1)

        # Encoding
        self.label_stoi = {
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
        }
        self.base_stoi = {
             "A": 7,
             "C": 8,
             "G": 9,
             "T": 10,
             "N": 11,
         }
        self.label_itos = {v:k for k, v in self.label_stoi.items()}
        self.base_itos = {v:k for k, v in self.base_stoi.items()}
        self.limit_itos = {0:7, 1:8, 2:9, 3:10}

        # Loss function
        self.loss = lambda logits, y: F.cross_entropy(logits, y, ignore_index=-1)

    def forward(self, x):
        logits = self.model(x)[0].logits.swapaxes(1,2)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.train_acc.update(logits.argmax(1), y)
        self.log(
            "train_loss",
            loss,
            logger=self.logger_type is not None,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.val_acc.update(logits.argmax(1), y)
        self.log(
            "val_loss",
            loss,
            logger=self.logger_type is not None,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_epoch_end(self, output):
        val_acc = self.val_acc.compute()
        val_loss = torch.mean(torch.Tensor(output))
        print(f"Val loss: {val_loss}, val acc: {val_acc}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.lr))
        return optimizer

    def count_params(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def logits_to_probs(self, logits):
        return F.softmax(logits, dim=1)

    def logits_to_idx(self, logits, sample=False):
        assert len(logits.shape) == 2

        # Get probabilities
        probs = self.logits_to_probs(logits)

        # Subset to valid bases
        probs = probs[:, [7, 8, 9, 10]]

        # Get next index based on limited bases
        if sample:
            limit_idx = torch.multinomial(probs, num_samples=1).squeeze().tolist()
        else:
            limit_idx = torch.topk(probs, k=1, dim=-1)[1].squeeze().tolist()

        # Convert to general index
        if isinstance(limit_idx, list):
            idx = [self.limit_itos[x] for x in limit_idx]
        else:
            idx = [self.limit_itos[limit_idx]]

        # Convert type
        idx = torch.Tensor(idx).type(torch.long)
        return idx.unsqueeze(0).to(logits.device)

    def train_on_dataset(self,
                         train_dataset, val_dataset,
                         batch_size=128, num_workers=8, device=0, max_epochs=3,
                         val_check_interval=5000,
                        ):
        torch.set_float32_matmul_precision("medium")

        # Save dataset
        self.seq_len = train_dataset.seq_len
        self.label_len = train_dataset.label_len

        # Logger
        if self.logger_type == "wandb":
            logger = WandbLogger(
                    name=datetime.now().strftime("%Y_%d_%m_%H_%M"),
                    log_model=True,
                    save_dir=self.save_dir,
                )
        elif self.logger_type == "csv":
            logger = CSVLogger(self.save_dir)
        else:
            logger = None

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=[device],
            logger=logger,
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min")],
            default_root_dir=self.save_dir,
            val_check_interval=val_check_interval,
        )

        # Make dataloaders
        train_data = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_data = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # First validation pass
        trainer.validate(model=self, dataloaders=val_data)

        # Training
        trainer.fit(model=self, train_dataloaders=train_data, val_dataloaders=val_data)

        return trainer

    def compute_accuracy_on_dataset(self, dataset, batch_size=64, num_workers=8, device=0,
                                    average="macro", multidim_average="global"):
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        acc = Accuracy(task="multiclass", num_classes=16, ignore_index=-1,
                       average=average, multidim_average=multidim_average).to(torch.device(device))
        for batch in iter(dl):
            x, y = batch
            x = x.to(torch.device(device))
            y = y.to(torch.device(device))
            logits = self.forward(x)
            y_hat = logits.argmax(1)
            acc.update(y_hat, y)

        return acc.compute().cpu().detach().numpy()

    def compute_positionwise_accuracy_on_dataset(self, dataset, batch_size=64, num_workers=8, device=0):
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        y_hat = []
        y = []
        
        for batch in iter(dl):
            x = batch[0].to(torch.device(device))
            logits = self.forward(x).squeeze()
            y_hat.append(logits.argmax(1).cpu().detach())
            y.append(batch[1].detach().squeeze())
        
        y_hat = torch.vstack(y_hat)
        y = torch.vstack(y)

        return (y_hat==y).numpy()

    def encode_label(self, label, add_start=False, add_batch_dim=False):
        idx = torch.tensor([self.label_stoi[tok] for tok in label], dtype=torch.long)
        if add_start:
            idx = torch.cat((torch.tensor([0], dtype=torch.long), idx))
        if add_batch_dim:
            idx = idx.unsqueeze(0)
        return idx

    def encode_seq(self, seq):
        return torch.tensor([self.base_stoi[tok] for tok in seq], dtype=torch.long)

    def encode(self, seq, label, add_start=False, add_stop=False, add_batch_dim=False):
        idx = torch.cat([self.encode_label(label), self.encode_seq(seq)])
        if add_start:
            idx = torch.cat((torch.tensor([0], dtype=torch.long), idx))
        if add_stop:
            idx = torch.cat((idx, torch.tensor([1], dtype=torch.long)))
        if add_batch_dim:
            idx = idx.unsqueeze(0)
        return idx

    def decode(self, ix):
        if len(ix.shape) == 2:
            ix = ix.squeeze()
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().detach().numpy()
        ix = ix[np.isin(ix, [7, 8, 9, 10, 11])]
        return "".join(self.base_itos[i] for i in ix)

    def P_seq(self, seq, log=True, per_pos=False, device="cpu"):
        idx = self.encode(seq, add_batch_dim=True, add_start=True)
        idx = idx.to(torch.device(device))
        logits = self.forward(idx)
        probs = self.logits_to_probs(logits).cpu().detach().numpy().squeeze()
        L = [probs[ix, pos] for ix, pos in zip(idx[0].tolist(), range(idx.shape[1]))]
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L)
            else:
                return np.product(L)

    def P_seq_given_label(self, seq, log=True, device="cpu"):
        L = self.P_seq(seq, log=log, per_pos=True, device=device)
        L = L[self.label_len:]
        if log:
            return np.sum(L)
        else:
            return np.product(L)

    def P_label_given_seq(self, seq, device="cpu"):
        label = seq[0]
        other_labels = np.setdiff1d(['A', 'C', 'G', 'T', 'N'], label).tolist()
        Ls = [self.P_seq_given_label(seq, log=False, device=device) for l in [label] + other_labels]
        LL = np.log(Ls[0])
        marginal_LL = np.log(np.sum(Ls))
        return LL - marginal_LL

    
    @torch.no_grad()
    def generate(self, label=None, max_new_tokens=None, temperature=1.0, sample=True, device="cpu"):

        # start sequence
        if label is None:
            idx = torch.zeros(1,1).type(torch.long)
        else:
            idx = self.encode_label(label, add_start=True, add_batch_dim=True)
    
        # Format input
        idx = idx.to(torch.device(device))

        # bases to add
        if max_new_tokens is None:
            max_new_tokens = self.seq_len + 1 - idx.shape[1]
        
        for _ in range(max_new_tokens):

            # Get logits
            logits = self.forward(idx)[:, :, -1]
        
            # scale by temperature
            logits = logits / temperature

            # Get next index
            idx_next = self.logits_to_idx(logits, sample=sample)
            if idx_next == 0: # stop token is produced
                break

            # Add new index
            idx = torch.cat((idx, idx_next), dim=1)

        return self.decode(idx)