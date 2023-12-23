import numpy as np

from reglm.dataset import CharDataset


def compute_accuracy(model, seqs, shuffle_labels=False, batch_size=64, num_workers=8):
    """
    Compute per-base accuracy of a trained regLM model on labeled sequences

    Args:
        model (pl.LightningModule): Trained regLM model
        seqs (pd.DataFrame): Dataframe containing sequences under 'Sequence'
            and labels under 'label'.
        shuffle_labels (bool): Whether to shuffle the labels among sequences
            before computing accuracy.
        batch_size (int): Batch size for inference
        num_workers (int): Number of workers for inference

    Returns:
        seqs (pd.DataFrame): original dataframe with added columns for per-
        base and average accuracy.
    """
    # Extract labels
    labels = seqs.label

    # Shuffle labels if needed
    if shuffle_labels:
        labels = seqs.label.sample(len(seqs))

    labels = labels.tolist()

    # Create dataset
    ds = CharDataset(seqs=seqs.Sequence.tolist(), labels=labels)

    # Compute per-base accuracy
    acc = model.compute_accuracy_on_dataset(
        ds, batch_size=batch_size, num_workers=num_workers
    )

    # Add results to dataframe
    if shuffle_labels:
        seqs["acc_shuf"] = acc
        seqs["acc_shuf_mean"] = seqs["acc_shuf"].apply(np.mean)
        avg_acc = seqs["acc_shuf_mean"].mean()
    else:
        seqs["acc"] = acc
        seqs["acc_mean"] = seqs["acc"].apply(np.mean)
        avg_acc = seqs["acc_mean"].mean()

    # Print overall mean
    print(f"Mean accuracy: {avg_acc:.3f}")

    return seqs
