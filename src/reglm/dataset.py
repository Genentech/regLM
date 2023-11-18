import numpy as np
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, seqs, labels, seq_len=None, rc=False, seed=0):
        """
        A dataset class to produce tokenized sequences for regLM.

        Args:
            seqs (list): List of sequences.
            labels (list): List of labels as strings
            seq_len (int): Maximum sequence length
            rc (bool): Augment sequence by reverse complementation
            seed (int): Random seed
        """
        # Check
        assert len(seqs) == len(labels), "seqs and labels should have equal length"
        assert (
            len(set([len(x) for x in labels])) == 1
        ), "All labels should be of equal length"

        # Store data
        self.seqs = seqs
        self.labels = labels
        self.rng = np.random.RandomState(seed)
        self.rc = rc

        # maximum sequence length
        if seq_len is None:
            seq_len = np.max([len(seq) for seq in self.seqs])
        self.seq_len = seq_len

        self.label_len = len(self.labels[0])
        self.output_len = self.seq_len + self.label_len + 2  # <START> label, seq, <End>
        self.n_unique_labels = len(
            set(np.concatenate([[tok for tok in lab] for lab in self.labels]))
        )
        assert (
            self.n_unique_labels <= 10
        ), ">10 label classes are currently not supported"

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
        self.rc_hash = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N",
        }
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq, is_labeled=False):
        """
        Encode a sequence as a torch tensor of tokens
        """
        if is_labeled:
            # Split the input into sequence and label
            label = seq[: self.label_len]
            seq = seq[self.label_len :]
            # Encode them separately and rejoin
            return torch.tensor(
                [self.label_stoi[tok] for tok in label]
                + [self.base_stoi[tok] for tok in seq],
                dtype=torch.long,
            )
        else:
            # Only a sequence is provided
            return torch.tensor([self.base_stoi[tok] for tok in seq], dtype=torch.long)

    def decode(self, ix, is_labeled=False):
        """
        Given a torch tensor of tokens, return the decoded sequence as a string.
        """
        if is_labeled:
            # Split the input into sequence and label
            label = ix[: self.label_len]
            seq = ix[self.label_len :]
            # Decode them separately and rejoin
            return "".join(
                [self.label_itos[i] for i in label] + [self.base_itos[i] for i in seq]
            )
        else:
            # Only a sequence is provided
            return "".join([self.base_itos[i] for i in ix])

    def __getitem__(self, idx):
        """
        Return a single labeled example as a tensor of tokens
        """
        # Get sequence
        seq = self.seqs[idx]

        # Reverse complement sequence if required
        if self.rc and self.rng.randint(2):
            seq = "".join([self.rc_hash[base] for base in reversed(seq)])

        # Get label
        label = self.labels[idx]

        # prefix the label to the sequence
        seq = label + seq

        # Encode label + sequence
        ix = self.encode(seq, is_labeled=True)

        # Generate empty tensors
        x = torch.zeros(self.output_len - 1, dtype=torch.long)
        y = torch.zeros(self.output_len - 1, dtype=torch.long)

        # Split sequence

        # Input: <START (0)>, label, sequence
        x[1 : 1 + len(ix)] = ix

        # Output: label, sequence, <END (0)>
        y[: len(ix)] = ix

        y[len(ix) + 1 :] = -1  # index -1 will mask the loss at the inactive locations
        return x, y
