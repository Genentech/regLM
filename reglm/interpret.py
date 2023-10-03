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
    seqs = np.random.choice(['A', 'C', 'G', 'T'], size=n * seq_len, replace=True).reshape(
        [n, seq_len]
    )

    return ["".join(seq) for seq in seqs]


def motif_likelihood(seq, motif, motif_idxs, label, model, device=0):
    
    # Append the motif to the sequence
    seq = seq + motif

    # Encode sequence with label 
    idx = model.encode(seq=seq, label=label, add_batch_dim=True, add_start=True).to(torch.device(device))
    n_tokens = idx.shape[1]

    # Make predictions
    logits = model.forward(idx)
    probs = model.logits_to_probs(logits).cpu().detach().numpy().squeeze()

    # Get predicted probabilities for the motif
    probs = probs[:, -(len(motif) + 1):-1]

    # Get likelihood of motif bases
    likelihood_per_pos = [probs[ix, pos] for ix, pos in zip(motif_idxs, range(n_tokens))]
    log_likelihood_per_pos = np.log(likelihood_per_pos)
    
    return log_likelihood_per_pos.sum()


def motif_insert(pwms, model, label, ref_label, n=100, seq_len=100):
    out = []
    random_seqs = generate_random_sequences(n=n, seq_len=seq_len)

    for seq in random_seqs:
        for row in pwms.iterrows():
            
            # Get the motif name
            motif_id = row[0]
            
            # Get the consensus sequence
            consensus = row[1].consensus
            motif_len = len(consensus)
            
            # Get the indices corresponding to the motif
            motif_tokens = model.encode_seq(consensus).numpy().tolist()

            # Compute log-likelihood with token 00 and 44
            LL_with_label = motif_likelihood(seq, consensus, motif_tokens, label, model)
            LL_with_ref = motif_likelihood(seq, consensus, motif_tokens, ref_label, model)

            # Compute log-likelihood ratio
            ratio = LL_with_label - LL_with_ref
            out.append([seq, motif_id, ratio])
    
    out = pd.DataFrame(out)
    out.columns=['Sequence', 'Motif', 'LL_ratio']
    return out