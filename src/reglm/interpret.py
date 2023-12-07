import numpy as np
import pandas as pd


def generate_random_sequences(n=1, seq_len=1024, seed=0):
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
        label (list): Labels
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


def motif_insert(pwms, model, label, ref_label, n=100, seq_len=100):
    """
    Insert motifs into random sequences and calculate log-likelihood ratio
    of each motif given label vs. reference label.

    Args:
        pwms (list): Sequences
        model (pl.LightningModule): regLM model
        label (list): Labels
        ref_label (str):
        n (int):
        seq_len (int):

    Returns:
        (list): log-likelihoods
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
