import numpy as np
import torch

from reglm.dataset import CharDataset


def test_chardataset():
    ds = CharDataset(["AAA", "AAC"], ["00", "01"], seq_len=4, rc=True, seed=0)
    assert np.all(ds.seqs == ["AAA", "AAC"])
    assert np.all(ds.labels == ["00", "01"])
    assert ds.rc
    assert ds.seq_len == 4
    assert ds.label_len == 2
    assert ds.n_unique_labels == 2
    assert torch.all(ds.encode_seq("ACGT") == torch.LongTensor([7, 8, 9, 10]))
    assert torch.all(ds.encode_label("10") == torch.LongTensor([3, 2]))
    assert ds.decode(torch.LongTensor([7, 8, 9, 10]), is_labeled=False) == "ACGT"
    assert ds.decode(torch.LongTensor([3, 2, 7, 8, 9, 10]), is_labeled=True) == "10ACGT"

    x, y = ds[0]
    assert torch.all(x == torch.LongTensor([2, 2, 7, 7, 7, 0]))
    assert torch.all(y == torch.LongTensor([7, 7, 7, 1, 0]))

    x, y = ds[1]
    assert torch.all(x == torch.LongTensor([2, 3, 9, 10, 10, 0]))
    assert torch.all(y == torch.LongTensor([9, 10, 10, 1, 0]))
