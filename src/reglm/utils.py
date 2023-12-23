import numpy as np

BASE_TO_IDX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def get_percentiles(values, n_bins=None, qlist=None):
    """
    Return list of tokens for sequences by binning their associated values

    Args:
        values (list): Values for which to calculate percentiles
        n_bins (int): Number of equal bins into which to split values
        qlist (list): Quantiles to split values into

    Returns:
        List containing percentiles at which to split the values
    """
    # If given a number of bins, split the given values into equal bins.
    if n_bins is not None:
        assert n_bins < len(values)
        binwidth = 100 / n_bins
        qlist = np.arange(binwidth, 100, binwidth)

    # Find values that split the values by percentiles
    return np.percentile(values, qlist)


def get_label_tokens(values, percentiles):
    """
    Return labels for sequences given cutoff percentiles

    Args:
        values (list): Values for which to calculate percentiles
        percentiles (list): Percentiles at which to split values

    Returns:
        list containing label token corresponding to each value
    """
    return [str(x) for x in np.digitize(values, percentiles)]


def tokenize(df, cols, names, n_bins=None, qlist=None, percentiles=None):
    """
    Create labels for sequences by dividing them into bins

    Args:
        df (pd.DataFrame): Dataframe containing label values
        cols (list): Names of columns to tokenize
        names (list): Names to use for the returned tokens
        n_bins (int): Number of equal bins into which to split values
        qlist (list): Quantiles to split values into
        percentiles (dict): Dictionary containing columns from cols
            as keys, and lists of percentile values.

    Returns:
        df (pd.DataFrame): Original dataframe with additional columns containing
        tokenized labels
    """
    # Get percentiles
    if percentiles is None:
        percentiles = dict()
        for col in cols:
            percentiles[col] = get_percentiles(df[col], n_bins=n_bins, qlist=qlist)
            print(col, percentiles[col].tolist())

    # Add a column to contain the label
    df["label"] = [""] * len(df)

    # Fill in tokens and labels
    for name, col in zip(names, cols):
        df[name + "_token"] = get_label_tokens(df[col], percentiles[col])
        df["label"] = df["label"] + df[name + "_token"]

    return df


def seqs_to_idxs(seqs):
    """
    Convert DNA sequences to indices

    Args:
        seqs (list): List of sequences to convert into indices

    Returns:
        np.array of shape (len(seqs), seq_len) containing the sequences
        as indices
    """
    return np.array([[BASE_TO_IDX[base] for base in seq] for seq in seqs])


def scores_to_matrix(scores, seqs):
    """
    Convert per-base scores to a N x seq_len x 4 numpy array

    Args:
        scores (torch.Tensor): tensor of shape N x seq_len
        seqs (list): List of DNA sequences of length N

    Returns:
        matrix (np.array): An array of shape N x seq_len x 4, in
            which the entries corresponding to each base in seqs
            will be filled with the values in scores, and other
            entries will be 0.
    """
    # Check shapes
    assert len(seqs) == scores.shape[0]

    # Encode sequences
    idxs = seqs_to_idxs(seqs)  # N, seq_len

    # Create empty array
    matrix = np.zeros((idxs.shape[0], idxs.shape[1], 4))  # N, seq_len, 4

    # Fill in empty matrix with scores
    for seq_idx in range(idxs.shape[0]):
        for pos in range(idxs.shape[1]):
            true_base_idx = idxs[seq_idx, pos]
            true_base_score = scores[seq_idx, pos].tolist()
            matrix[seq_idx, pos, true_base_idx] = true_base_score

    return matrix


def matrix_to_scores(matrix, seqs):
    """
    Convert a tensor of shape N x seq_len 4 to a 2-D array of shape N, seq_len
    containing scores for the actual bases in each sequence

    Args:
        matrix (torch.Tensor): An tensor of shape N x seq_len x 4
        seqs (list): List of DNA sequences of length N

    Returns:
        scores (np.array): array of shape N x seq_len, which will contain
            the values in matrix that correspond to the real bases in seqs.
    """
    # Encode sequences
    idxs = seqs_to_idxs(seqs)

    # Create empty array
    scores = np.zeros(idxs.shape)

    # Fill the empty array with scores of the true base
    for seq_idx in range(idxs.shape[0]):
        for pos in range(idxs.shape[1]):
            true_base_idx = idxs[seq_idx, pos]
            true_base_score = matrix[seq_idx, pos, true_base_idx].tolist()
            scores[seq_idx, pos] = true_base_score

    return scores
