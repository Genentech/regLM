import numpy as np
import pandas as pd
import torch


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
    np.random.seed(seed)

    # Generate sequences
    seqs = np.random.choice(
        ["A", "C", "G", "T"], size=n * seq_len, replace=True
    ).reshape([n, seq_len])

    return ["".join(seq) for seq in seqs]


def motif_likelihood(seqs, motif, label, model, device=0):
    log_likelihood_per_pos = model.P_seqs_given_labels([seq+motif for seq in seqs], labels=[label]*len(seqs), per_pos=True, log=True, device=device)
    return log_likelihood_per_pos[:, -(len(motif)+1):-1].sum(1)


def motif_insert(pwms, model, label, ref_label, n=100, seq_len=100):
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
        curr_out = pd.DataFrame({
            "Sequence":random_seqs,
            "Motif": motif_id,
            "LL_ratio": ratio,
        })
        out = pd.concat([out, curr_out])

    return out
