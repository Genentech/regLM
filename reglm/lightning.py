import json
import os
import sys
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy

sys.path.append("/code/hyena-dna")
from src.models.sequence.long_conv_lm import ConvLMHeadModel


def load_pretrained_model(ckpt_dir="./checkpoints/", model="hyenadna-tiny-1k-seqlen"):
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
        os.system(
            f"wget -P {ckpt_dir} https://huggingface.co/LongSafari/{model}/resolve/main/config.json"
        )
        os.system(
            f"wget -P {ckpt_dir} https://huggingface.co/LongSafari/{model}/resolve/main/weights.ckpt"
        )

    # Load config
    config = json.load(open(os.path.join(ckpt_dir, "config.json"), "r"))

    # Generate model
    model = ConvLMHeadModel(**config)

    # Load weights
    state_dict = torch.load(
        os.path.join(ckpt_dir, "weights.ckpt"), map_location=torch.device("cpu")
    )
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
            raise ValueError("either config or ckpt_dir must be provided.")
        n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

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
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}

        # Loss function
        self.loss = lambda logits, y: F.cross_entropy(logits, y, ignore_index=-1)
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape (N, 16, L)
        """
        return self.model(x)[0].logits.swapaxes(1, 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
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
        self.log("val_acc", val_acc)
        val_loss = torch.mean(torch.Tensor(output))
        print(f"Val loss: {val_loss}, val acc: {val_acc}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.lr))
        return optimizer        

    def train_on_dataset(
        self,
        train_dataset,
        val_dataset,
        batch_size=128,
        num_workers=8,
        device=0,
        max_epochs=3,
        val_check_interval=5000,
        weights=None,
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
            callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
            default_root_dir=self.save_dir,
            val_check_interval=val_check_interval,
        )

        # Make dataloaders
        if weights is None:
            train_data = DataLoader(
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
            train_data = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=sampler,
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

    def compute_accuracy_on_dataset(
        self, dataset, batch_size=64, num_workers=8, device=0
    ):
        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        y_hat = []
        y = []

        for batch in iter(dl):
            x = batch[0].to(torch.device(device))
            logits = self.forward(x).squeeze()
            y_hat.append(logits.argmax(1).cpu().detach())
            y.append(batch[1].detach().squeeze())

        y_hat = torch.vstack(y_hat)
        y = torch.vstack(y)

        return (y_hat == y).numpy()

    def encode_labels(self, labels, add_start=False):
        """
        Args:
            label (list, str): Strings of label tokens
            add_start (bool): Add a start token (0) before the label

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L) or (N, L+1)
        """
        if isinstance(labels, str):
            labels = [labels]
        idxs = torch.LongTensor([[self.label_stoi[tok] for tok in label] for label in labels]) # N, L
        if add_start:
            idxs = torch.cat((torch.LongTensor([[0]] * idxs.shape[0]), idxs), axis=1) # N, L+1
        return idxs

    def encode_seqs(self, seqs, add_stop=False):
        """
        Args:
            seqs (list, str): Strings of base tokens
            add_stop (bool): Add an end token (0) after the sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L) or (N, L+1)
        """
        if isinstance(seqs, str):
            seqs = [seqs]

        idxs = torch.LongTensor([[self.base_stoi[tok] for tok in seq] for seq in seqs]) # N, L
        if add_stop:
            idxs = torch.cat((idxs, torch.LongTensor([[0]] * idxs.shape[0])), axis=1) # N, L+1
        return idxs

    def encode(self, seqs, labels, add_start=False, add_stop=False):
        """
        Args:
            seqs (list, str): Strings of base tokens
            label (list, str): Strings of label tokens
            add_start (bool): Add a start token (0) before the label
            add_stop (bool): Add an end token (0) after the sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        return torch.cat([self.encode_labels(labels, add_start=add_start), self.encode_seqs(seqs, add_stop=add_stop)], axis=1)

    def decode(self, idxs):
        """
        Args:
            idxs (torch.LongTensor, np.array): tensor or array of shape (N, L)

        Returns:
            seqs (list): list of strings
        """
        if idxs.dim() == 1: # L
            idxs = idxs.unsqueeze(0) # N, L
        
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu().detach().numpy()

        # Replace non-bases with N
        idxs[idxs < 7] = 11
        idxs[idxs > 11] = 11

        seqs = []
        
        for seq_idxs in idxs:
            curr_seq = []
            for ix in seq_idxs:
                if (ix == 0) and (pos > 0):
                    break
                curr_seq.append(self.base_itos[ix])
            seqs.append("".join(curr_seq))
        return seqs

    def logits_to_probs(self, logits):
        """
        Args:
            logits (torch.tensor, dtype torch.float32): tensor of shape (N, 16, L) or (N, 16)

        Returns:
            tensor of shape (N, 16, L) or (N, 16)
        """
        assert logits.dim() in [2, 3]
        assert logits.shape[1] == 16
        return logits.softmax(1)

    def probs_to_likelihood(self, idxs, probs):
        """
        Args:
            probs (torch.FloatTensor): tensor of shape (N, 16, L)
            idxs (torch.LongTensor): tensor of shape (N, L)

        Returns:
            tensor of shape (N, L)
        """
        mask = F.one_hot(idxs, num_classes=16).type(torch.bool)
        return torch.masked_select(probs.swapaxes(1, 2).cpu().detach(), mask).reshape(idxs.shape)
        
    def P_seqs(self, seqs, labels, log=True, per_pos=False, device="cpu"):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            per_pos (bool): Return likelihood for each base
            device (str, int): device index

        Returns:
            np.array of shape (N, L) or (N)
        """
        idxs = self.encode(seqs, labels, add_start=True, add_stop=True) # N, L+2
        logits = self.forward(idxs[:, :-1].to(torch.device(device))) # N, 16, L+1
        probs = self.logits_to_probs(logits) # N, 16, L+1
        L = self.probs_to_likelihood(idxs[:, 1:], probs).numpy() # N, L+1
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1) # N
            else:
                return np.product(L, 1) # N

    def P_seqs_given_labels(self, seqs, labels, per_pos=False, log=True, device="cpu"):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            device (str, int): device index

        Returns:
            np.array of shape (N)
        """
        L = self.P_seqs(seqs, labels, log=log, per_pos=True, device=device) # N, L
        L = L[:, self.label_len :] # N, seq_len
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1) # N
            else:
                return np.product(L,1) # N

    def filter_base_probs(self, probs):
        """
        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)

        Returns:
            tensor of shape (N, 4)
        """
        assert probs.dim() == 2
        assert probs.shape[1] == 16
        return probs[:, [7,8,9,10]]

    def threshold_probs(self, filtered_probs, top_k=None, top_p=None):
        """
        Args:
            filtered_probs (torch.tensor, dtype torch.float32): tensor of shape (N, 4)

        Returns:
            tensor of shape (N, 4)
        """
        assert filtered_probs.dim() == 2
        assert filtered_probs.shape[1] == 4

        if top_k is not None:
            p_idxs = filtered_probs.argsort(1, descending=True)
            for seq_idx in range(filtered_probs.shape[0]):
                filtered_probs[seq_idx, p_idxs[seq_idx, top_k:]] = 0
            
        if top_p is not None:
            p_sorted, p_idxs = filtered_probs.sort(1, descending=True)
            cut = (p_sorted.cumsum(1) > top_p).cpu().detach().numpy().argmax(1).tolist()
            for seq_idx, cut_idx  in enumerate(cut):
                if cut_idx < 3:
                    filtered_probs[seq_idx, p_idxs[seq_idx, cut_idx+1:]] = 0

        return filtered_probs

    def logits_to_idxs(self, logits, top_k=None, top_p=None):
        """
        Args:
            logits (torch.tensor, dtype torch.float32): tensor of shape (N, 16)

        Returns:
            idxs (torch.LongTensor): tensor of shape (N)
        """
        assert logits.dim() == 2
        assert logits.shape[1] == 16

        # Get probabilities
        probs = self.logits_to_probs(logits) # N, 16

        # Subset to valid bases
        probs = self.filter_base_probs(probs) # N, 4

        # Filter probs
        probs = self.threshold_probs(probs, top_k=top_k, top_p=top_p) # N, 4

        # Re-normalize
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Sample
        idxs = probs.multinomial(1).squeeze() + 7

        # Send to device
        return idxs.to(logits.device)

    @torch.no_grad()
    def generate(
        self,
        labels,
        max_new_tokens=None,
        temperature=1.0,
        device="cpu",
        top_k=None,
        top_p=None,
    ):
        """
        Args:
            labels (str, list): Strings of label tokens
            max_new_tokens (int): Maximum number of tokens to add
            temperature (float): Temperature
            device (str, int): Device index
            top_k (int): Sample from top k bases
            top_p (float): Sample from bases with top_p cumulative probability
            
        Returns:
            seqs (list): List of strings
        """
        # Check labels
        if isinstance(labels, str):
            labels = [labels]
        assert len(labels[0]) == self.label_len

        # bases to add
        if max_new_tokens is None:
            max_new_tokens = self.seq_len

        # Encode labels
        idxs = self.encode_labels(labels, add_start=True).to(torch.device(device)) # N, L+1

        # Add bases
        for _ in range(max_new_tokens):

            # Get logits
            logits = self.forward(idxs)[:, :, -1] # N, 16

            if temperature != 1.:
                # scale by temperature
                logits = logits / temperature

            # Get next indices
            idxs_next = self.logits_to_idxs(logits, top_k=top_k, top_p=top_p) # N

            # Add new indices
            idxs = torch.cat((idxs, idxs_next.unsqueeze(1)), dim=1)

        return self.decode(idxs[:, self.label_len+1:])
