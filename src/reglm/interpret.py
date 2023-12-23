import numpy as np
import pandas as pd

from reglm.regression import SeqDataset
from reglm.utils import seqs_to_idxs


def ISM_at_pos(seq, pos, drop_ref=True):
    """
    Perform in-silico mutagenesis at a single position in the sequence.

    Args:
        seq (str): DNA sequence
        pos (int): Position to mutate
        drop_ref (bool): If True, the original base at the mutation position is dropped.

    Returns:
        List of mutated DNA sequences, of length 3 or 4
    """
    alt_bases = ["A", "C", "G", "T"]
    if drop_ref:
        alt_bases.remove(seq[pos])

    return [seq[:pos] + base + seq[pos + 1 :] for base in alt_bases]


def ISM(seq, drop_ref=True):
    """
    Perform in-silico mutagenesis of a DNA sequence.

    Args:
        seq (str): DNA sequence
        drop_ref (bool): If True, the original base at the mutation position is dropped.

    Returns:
        List of mutated DNA sequences, of length 3*len(seq) or 4*len(seq)
    """
    return list(
        np.concatenate(
            [ISM_at_pos(seq, pos, drop_ref=drop_ref) for pos in range(len(seq))]
        )
    )


def ISM_predict(seqs, model, seq_len=None, batch_size=512, device=0):
    """
    Perform in-silico mutagenesis of DNA sequences and make predictions with a
    regression model to get per-base importance scores

    Args:
        seqs (list): List of DNA sequences of equal length
        model (pl.LightningModule): regression model
        seq_len (int): Maximum sequence length for regression model
        batch_size (int): Batch size for prediction
        device (int): GPU index for prediction

    Returns:
        np.array of shape (number of sequences x length of sequences) or
        (number of sequences x length of sequences x number of tasks), containing
        per-base importance scores
    """

    # Get sequence length
    actual_seq_lens = [len(seq) for seq in seqs]
    assert (
        len(set(actual_seq_lens)) == 1
    ), "This function currently requires all sequences to have equal length"
    actual_seq_len = actual_seq_lens[0]

    # Perform ISM
    mutated_seqs = np.concatenate([ISM(seq, drop_ref=False) for seq in seqs])  # N*4

    # Get predictions for all mutated sequences
    dataset = SeqDataset(mutated_seqs, seq_len=seq_len)
    preds = model.predict_on_dataset(
        dataset,
        device=device,
        batch_size=batch_size,
    )  # Nx4xseq_len, n_tasks

    # Reshape the predictions
    if preds.ndim == 1:
        preds = np.expand_dims(preds, 1)
    assert (preds.ndim == 2) and (
        preds.shape[0] == len(seqs) * 4 * actual_seq_len
    ), preds.shape

    preds = preds.reshape(
        len(seqs), 4 * actual_seq_len, preds.shape[-1]
    )  # N, 4*seq_len, n_tasks

    preds = preds.reshape(
        len(seqs), actual_seq_len, 4, preds.shape[-1]
    )  # N, seq_len, 4, n_tasks

    # Make empty array for final scores
    scores = np.zeros(np.delete(preds.shape, 2))  # N, seq_len, n_tasks

    # Convert original sequences to indices
    idxs = seqs_to_idxs(seqs)  # N, seq_len

    # Take the log-ratio of the predicted scores relative to the original sequence
    for seq_idx in range(scores.shape[0]):
        for pos_idx in range(scores.shape[1]):
            # Get predictions at true and mutated bases
            true_idx = idxs[seq_idx, pos_idx]
            mutated_idxs = [idx for idx in range(4) if idx != true_idx]
            true_preds = preds[seq_idx, pos_idx, true_idx]  # n_tasks
            mutated_preds = preds[seq_idx, pos_idx, mutated_idxs]  # 3, n_tasks

            # Calculate average log2 ratio caused by mutation
            score = -np.log2(mutated_preds / np.expand_dims(true_preds, 0)).mean(
                0
            )  # n_tasks

            # Insert
            scores[seq_idx, pos_idx] = score

    return scores.squeeze()  # N, seq_len, n_tasks


def generate_random_sequences(n=1, seq_len=1024, seed=None):
    """
    Generate random DNA sequences.

    Args:
        n (int): Number of sequences to generate (default 1).
        seq_len (int): Length of each sequence (default 1024).
        seed (int): Seed value for random number generator (default 0).

    Returns:
        Generated sequences as a list of strings.
    """
    # Set random seed
    rng = np.random.RandomState(seed)

    # Generate sequences
    seqs = rng.choice(["A", "C", "G", "T"], size=n * seq_len, replace=True).reshape(
        [n, seq_len]
    )

    return ["".join(seq) for seq in seqs]


def motif_likelihood(seqs, motif, label, model):
    """
    Return the log-likelihood of a motif occurring at the end of
    each of the given sequences.

    Args:
        seqs (list): Sequences
        motif (seq): Motif sequence
        label (list): Label for the regLM model
        model (pl.LightningModule): regLM model

    Returns:
        (list): log-likelihoods
    """
    log_likelihood_per_pos = model.P_seqs_given_labels(
        seqs=[seq + motif for seq in seqs],
        labels=[label] * len(seqs),
        per_pos=True,
        log=True,
    )
    motif_likelihood = log_likelihood_per_pos[:, -len(motif) :]
    assert motif_likelihood.shape == (len(seqs), len(motif)), motif_likelihood.shape
    return motif_likelihood.sum(1)


def motif_insert(pwms, model, label, ref_label, seq_len, n=100):
    """
    Insert motifs into random sequences and calculate log-likelihood ratio
    of each motif given label vs. reference label.

    Args:
        pwms (list): Sequences
        model (pl.LightningModule): regLM model
        label (list): Label for the regLM model
        ref_label (str):
        seq_len (int): Length of random sequences preceding the motif
        n (int): number of random sequences to insert the motif in

    Returns:
        (pd.DataFrame): Dataframe containing log likelihood ratios of motif-containing
        sequences
    """
    out = pd.DataFrame()
    random_seqs = generate_random_sequences(n=n, seq_len=seq_len)

    for row in pwms.iterrows():
        # Get the motif name
        motif_id = row[0]

        # Get the consensus sequence
        consensus = row[1].consensus

        # Compute log-likelihood with token 00 and 44
        LL_with_label = motif_likelihood(random_seqs, consensus, label, model)
        LL_with_ref = motif_likelihood(random_seqs, consensus, ref_label, model)

        # Compute log-likelihood ratio
        ratio = LL_with_label - LL_with_ref
        curr_out = pd.DataFrame(
            {
                "Sequence": random_seqs,
                "Motif": motif_id,
                "LL_ratio": ratio,
            }
        )
        out = pd.concat([out, curr_out])

    return out
