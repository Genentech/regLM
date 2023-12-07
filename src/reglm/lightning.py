import itertools
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


def load_pretrained_model(
    ckpt_dir="./checkpoints/",
    model="hyenadna-tiny-1k-seqlen",
    hyenadna_path="/code/hyena-dna",
):
    sys.path.append(hyenadna_path)
    from src.models.sequence.long_conv_lm import ConvLMHeadModel

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
        config = f"https://huggingface.co/LongSafari/{model}/resolve/main/config.json"
        ckpt = f"https://huggingface.co/LongSafari/{model}/resolve/main/weights.ckpt"
        os.system(f"wget -P {ckpt_dir} {config}")
        os.system(f"wget -P {ckpt_dir} {ckpt}")

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
    def __init__(
        self,
        config=None,
        ckpt_dir=None,
        logger=None,
        save_dir=".",
        lr=1e-4,
        label_len=None,
        hyenadna_path="/code/hyena-dna",
    ):
        super().__init__()
        sys.path.append(hyenadna_path)
        from src.models.sequence.long_conv_lm import ConvLMHeadModel

        self.save_dir = save_dir
        self.label_len = label_len
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr

        # Build model
        if ckpt_dir is not None:
            self.model = load_pretrained_model(ckpt_dir, hyenadna_path=hyenadna_path)
        elif config is not None:
            sys.path.append(hyenadna_path)
            self.model = ConvLMHeadModel(**config)
        else:
            raise ValueError("either config or ckpt_dir must be provided.")
        n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

        # Logger
        self.logger_type = logger

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)
        self.val_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)

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
        # Trailing zeros in the label will be ignored in calculating the loss
        self.loss = lambda logits, y: F.cross_entropy(logits, y, ignore_index=0)

    def forward(self, x, drop_label=True):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len - 1)
        """
        logits = self.model(x)[0].logits.swapaxes(
            1, 2
        )  # N, label + seq + end + trailing zeros

        # Drop the label probabilities
        if drop_label:
            logits = logits[:, :, self.label_len :]  # N, seq + end + trailing

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x, drop_label=True)  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
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
        logits = self.forward(x, drop_label=True)  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
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
        print(f"\nVal loss: {val_loss}, val acc: {val_acc}")

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

        # Save dataset params
        self.seq_len = train_dataset.seq_len
        self.label_len = train_dataset.label_len
        self.unique_labels = train_dataset.unique_labels

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
        self,
        dataset,
        batch_size=64,
        num_workers=8,
    ):
        """
        Return per-example accuracy
        Note: this will include the accuracy of predicting the END token (1)
        """

        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        y_hat = []
        y = []

        self.eval()
        for batch in iter(dl):
            x = batch[0].to(self.device)
            logits = self.forward(
                x, drop_label=True
            ).squeeze()  # N, 16, seq+end+trailing
            y_hat.append(logits.argmax(1).cpu().detach())  # N, seq+end+trailing
            y.append(batch[1].detach().squeeze())  # N, seq+end+trailing zeros

        y_hat = torch.vstack(y_hat).numpy()  # N, L
        y = torch.vstack(y).numpy()  # N, L

        is_non_zero = y != 0
        is_equal = y_hat == y
        return [e[n] for e, n in zip(is_equal, is_non_zero)]

    def encode_labels(self, labels, add_start=False):
        """
        Args:
            label (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)

        Returns:
            (torch.LongTensor): tensor of shape (N, L)
        """
        if isinstance(labels, str):
            labels = [labels]
        idxs = torch.LongTensor(
            [[self.label_stoi[tok] for tok in label] for label in labels]
        )  # N, label_len
        if add_start:
            idxs = torch.cat(
                (torch.LongTensor([[0]] * idxs.shape[0]), idxs), axis=1
            )  # N, 1 + label_len
        return idxs

    def encode_seqs(self, seqs, add_stop=False):
        """
        Args:
            seqs (list, str): Strings of base tokens
            add_stop (bool): Add an end token (1) after the sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L) or (N, L+1)
        """
        if isinstance(seqs, str):
            seqs = [seqs]

        idxs = torch.LongTensor(
            [[self.base_stoi[tok] for tok in seq] for seq in seqs]
        )  # N, seq_len
        if add_stop:
            idxs = torch.cat(
                (idxs, torch.LongTensor([[1]] * idxs.shape[0])), axis=1
            )  # N, seq_len+1
        return idxs

    def encode(self, seqs, labels, add_start=False, add_stop=False):
        """
        Args:
            seqs (list, str): Strings of base tokens
            label (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)
            add_stop (bool): Add an end token (1) after the sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        return torch.cat(
            [
                self.encode_labels(labels, add_start=add_start),
                self.encode_seqs(seqs, add_stop=add_stop),
            ],
            axis=1,
        )

    def decode(self, idxs):
        """
        Args:
            idxs (torch.LongTensor, np.array): tensor or array of shape (N, L)

        Returns:
            seqs (list): list of strings
        """
        if idxs.dim() == 1:  # L
            idxs = idxs.unsqueeze(0)  # 1, L

        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu().detach().numpy()

        # Create empty list
        seqs = []

        # Fill list
        for x in idxs:
            curr_seq = []

            # Decode
            for pos, ix in enumerate(x):
                # Ignore start token
                if (pos == 0) and (ix == 0):
                    continue

                # Terminate at end token
                if ix == 1:
                    break

                # Replace non-bases with N
                elif (ix < 7) or (ix > 11):
                    ix = 11

                curr_seq.append(self.base_itos[ix])

            seqs.append("".join(curr_seq))
        return seqs

    def logits_to_probs(self, logits):
        """
        Args:
            logits (torch.tensor, dtype torch.float32): tensor of shape (N, 16, L)
                or (N, 16)

        Returns:
            tensor of shape (N, 16, L) or (N, 16)
        """
        assert logits.dim() in [2, 3]
        assert logits.shape[1] == 16
        return logits.softmax(1)

    def probs_to_likelihood(self, probs, idxs):
        """
        Args:
            probs (torch.FloatTensor): tensor of shape (N, 16, L)
            idxs (torch.LongTensor): tensor of shape (N, L)

        Returns:
            tensor of shape (N, L)
        """
        assert probs.dim() == 3, probs.shape
        assert probs.shape[1] == 16, probs.shape
        assert idxs.dim() == 2, idxs.shape
        assert idxs.shape[0] == probs.shape[0], (idxs.shape, probs.shape)
        assert idxs.shape[1] == probs.shape[2], (idxs.shape, probs.shape)

        mask = F.one_hot(idxs, num_classes=16).type(torch.bool)
        return torch.masked_select(probs.swapaxes(1, 2).cpu().detach(), mask).reshape(
            idxs.shape
        )

    def P_seqs(self, seqs, labels, per_pos=False, log=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=True
        )  # N, 0+label+seq+1

        # Compute probabilities
        self.eval()
        logits = self.forward(idxs.to(self.device), drop_label=False)[
            :, :, :-1
        ]  # N, 16, label+seq+1
        probs = self.logits_to_probs(logits)  # N, 16, label+seq+1

        # Compute likelihood
        idxs = idxs[:, 1:]  # N, label+seq+1
        L = self.probs_to_likelihood(probs, idxs).numpy()  # N, label+seq+1

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_seqs_given_labels(self, seqs, labels, per_pos=False, log=True, add_stop=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        # Encode sequence with labels
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=add_stop
        )  # N, 0+label+seq

        # Compute probabilities
        self.eval()
        logits = self.forward(idxs.to(self.device), drop_label=True)[
            :, :, :-1
        ]  # N, 16, seq
        probs = self.logits_to_probs(logits)  # N, 16, seq
        idxs = idxs[:, 1 + self.label_len :]  # N, seq

        # Compute likelihoods
        L = self.probs_to_likelihood(probs=probs, idxs=idxs).numpy()  # N, seq

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_labels_given_seqs(self, seqs, labels, per_pos=True, log=True):
        possible_labels = [
            "".join(x)
            for x in itertools.permutations(self.unique_labels, self.label_len)
        ]
        likelihoods = np.stack(
            [
                self.P_seqs(
                    seqs,
                    [label] * len(seqs),
                    per_pos=True,
                    log=False,
                )
                for label in possible_labels
            ],
            axis=-1,
        )  # N, L, possible_labels
        denominators = np.sum(likelihoods, -1)  # N, L
        numerators = [
            lik[:, possible_labels == label] for lik, label in zip(likelihoods, labels)
        ]  # N, L
        if log:
            P = np.log(numerators) - np.log(denominators)  # N, L
        else:
            P = np.divide(numerators, denominators)  # N, L
        if per_pos:
            return P
        else:
            if log:
                return np.sum(P, 1)  # N
            else:
                return np.product(P, 1)  # N

    def normalize_filtered_probs(self, filtered_probs):
        return filtered_probs / filtered_probs.sum(dim=1, keepdim=True)

    def filter_base_probs(self, probs, normalize=True):
        """
        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)

        Returns:
            tensor of shape (N, 4)
        """
        assert probs.dim() == 2
        assert probs.shape[1] == 16
        filtered_probs = probs[:, [7, 8, 9, 10]]
        if normalize:
            filtered_probs = self.normalize_filtered_probs(filtered_probs)
        return filtered_probs

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
            for seq_idx, cut_idx in enumerate(cut):
                if cut_idx < 3:
                    filtered_probs[seq_idx, p_idxs[seq_idx, cut_idx + 1 :]] = 0

        return filtered_probs

    def logits_to_idxs(
        self, logits, random_state=None, top_k=None, top_p=None, normalize_filtered=True
    ):
        """
        Args:
            logits (torch.tensor, dtype torch.float32): tensor of shape (N, 16)

        Returns:
            idxs (torch.LongTensor): tensor of shape (N)
        """
        assert logits.dim() == 2, logits.shape
        assert logits.shape[1] == 16, logits.shape

        # Get probabilities
        probs = self.logits_to_probs(logits)  # N, 16

        # Subset to valid bases
        probs = self.filter_base_probs(probs, normalize=normalize_filtered)  # N, 4

        # Filter probs
        probs = self.threshold_probs(probs, top_k=top_k, top_p=top_p)  # N, 4

        # Re-normalize
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Sample
        if random_state is None:
            idxs = probs.multinomial(1).squeeze() + 7
        else:
            idxs = probs.multinomial(1, generator=random_state).squeeze() + 7

        # Send to device
        return idxs.to(logits.device)

    @torch.no_grad()
    def generate(
        self,
        labels,
        max_new_tokens=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        normalize_filtered=True,
        seed=None,
    ):
        """
        Args:
            labels (str, list): Strings of label tokens
            max_new_tokens (int): Maximum number of tokens to add
            temperature (float): Temperature
            top_k (int): Sample from top k bases
            top_p (float): Sample from bases with top_p cumulative probability
            normalize_filtered (bool): Normalize probabilities to sum to 1
                after filtering

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
        idxs = self.encode_labels(labels).to(self.device)  # N, L+1

        # Get random state
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)

        # Add bases
        for _ in range(max_new_tokens):
            self.eval()
            # Get logits
            logits_next = self.forward(idxs)[:, :, -1]  # N, 16

            if temperature != 1.0:
                # scale by temperature
                logits_next = logits_next / temperature

            # Get next indices
            idxs_next = self.logits_to_idxs(
                logits_next,
                random_state=rng,
                top_k=top_k,
                top_p=top_p,
                normalize_filtered=normalize_filtered,
            )  # N

            # Add new indices
            idxs = torch.cat((idxs, idxs_next.unsqueeze(1)), dim=1)

        return self.decode(idxs[:, self.label_len :])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyper_parameters"]["seq_len"] = self.seq_len
        checkpoint["hyper_parameters"]["label_len"] = self.label_len
        checkpoint["hyper_parameters"]["unique_labels"] = self.unique_labels
