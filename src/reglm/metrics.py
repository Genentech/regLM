import numpy as np

import reglm.dataset


def compute_accuracy(model, seqs, shuffle_labels=False):
    labels = seqs.label
    if shuffle_labels:
        labels = seqs.label.sample(len(seqs))
    labels = labels.tolist()
    ds = reglm.dataset.CharDataset(seqs=seqs.Sequence.tolist(), labels=labels, rc=False)
    acc = model.compute_accuracy_on_dataset(ds, batch_size=64, num_workers=8)
    if shuffle_labels:
        seqs["acc_shuf"] = acc
        seqs["acc_shuf_mean"] = seqs["acc_shuf"].apply(np.mean)
        avg_acc = seqs["acc_shuf_mean"].mean()
    else:
        seqs["acc"] = acc
        seqs["acc_mean"] = seqs["acc"].apply(np.mean)
        avg_acc = seqs["acc_mean"].mean()
    print(f"Mean accuracy: {avg_acc:.3f}")
    return seqs
