import numpy as np
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, seqs, labels, seq_len=None):
        """
        A dataset class to produce tokenized sequences for training regLM.

        Each sequence will be represented as 0<LABEL><SEQ>1; hence 0 is the start
        token and 1 is the end token.

        Args:
            seqs (list): List of sequences.
            labels (list): List of labels as strings
            seq_len (int): Maximum sequence length
        """
        # Check
        assert len(seqs) == len(labels), "seqs and labels should have equal length"
        assert (
            len(set([len(x) for x in labels])) == 1
        ), "All labels should be of equal length"

        # Store data
        self.seqs = seqs
        self.labels = labels

        # maximum sequence length
        self.seq_len = seq_len or np.max([len(seq) for seq in self.seqs])
        self.label_len = len(self.labels[0])
        self.unique_labels = set(
            np.concatenate([[tok for tok in lab] for lab in self.labels])
        )
        assert (
            len(self.unique_labels) <= 10
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
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}

    def __len__(self):
        return len(self.seqs)

    def encode_seq(self, seq):
        """
        Encode a sequence as a torch tensor of tokens

        Args:
            seq (str): DNA sequence

        Returns:
            torch.LongTensor of shape (seq_len,)
        """
        return torch.LongTensor([self.base_stoi[tok] for tok in seq])

    def encode_label(self, label):
        """
        Encode a label as a torch tensor of tokens

        Args:
            label (str): label token sequence

        Returns:
            torch.LongTensor of shape (label_len,)
        """
        return torch.tensor([self.label_stoi[tok] for tok in label])

    def decode(self, idxs, is_labeled=False):
        """
        Given a torch tensor of tokens, return the decoded sequence as a string.

        Args:
            idxs (list, torch.LongTensor): list or 1-D tensor
            is_labeled (bool): Whether labels are included

        Returns:
            labeled sequence as a string
        """
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.detach().cpu().tolist()
        if is_labeled:
            # Split the input into sequence and label
            label = idxs[: self.label_len]
            seq = idxs[self.label_len :]
            # Decode them separately and rejoin
            return "".join(
                [self.label_itos[i] for i in label] + [self.base_itos[i] for i in seq]
            )
        else:
            # Only a sequence is provided
            return "".join([self.base_itos[i] for i in idxs])

    def __getitem__(self, idx):
        """
        Return a single labeled example as a tensor of tokens
        x = 0<LABEL><SEQ>
        y = <SEQ>1

        Args:
            idx: Index of example to return

        Returns:
            x (torch.LongTensor): tensor of shape (1 + self.label_len + self.seq_len)
            y (torch.LongTensor): tensor of shape (self.seq_len + 1, )
        """
        # Get sequence
        seq = self.seqs[idx]

        # Encode sequence
        seq = self.encode_seq(seq)

        # Get label
        label = self.labels[idx]

        # Encode label
        label = self.encode_label(label)

        # Generate empty tensors
        x = torch.zeros(self.seq_len + self.label_len + 1, dtype=torch.long)
        y = torch.zeros(self.seq_len + 1, dtype=torch.long)

        # Input: START(0) + label + sequence + trailing zeros (will be ignored)
        x[1 : 1 + self.label_len] = label
        x[1 + self.label_len : 1 + self.label_len + len(seq)] = seq

        # Output: sequence + END (1) + trailing zeros (will be ignored)
        y[: len(seq)] = seq
        y[len(seq)] = 1

        return x, y
