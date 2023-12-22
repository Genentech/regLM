import numpy as np
import pandas as pd


def get_percentiles(values, n_bins=None, qlist=None):
    """
    Return list of tokens for sequences by binning their associated values
    """
    # If given a number of bins, split the given values into equal bins.
    if n_bins is not None:
        assert n_bins < len(values)
        binwidth = 100 / n_bins
        qlist = np.arange(binwidth, 100, binwidth)

    # Find values that split the values by percentiles
    return np.percentile(values, qlist)


def get_labels(values, percentiles):
    """
    Return labels for sequences given cutoff percentiles
    """
    return [str(x) for x in np.digitize(values, percentiles)]


def tokenize(df, cols, names, n_bins, percentiles=None):
    """
    Create labels for sequences by dividing them into bins
    """
    if percentiles is None:
        percentiles = dict()
        for col in cols:
            percentiles[col] = get_percentiles(df[col], n_bins=n_bins)
            print(col, percentiles[col].tolist())

    df["label"] = [""] * len(df)

    for name, col in zip(names, cols):
        df[name + "_token"] = get_labels(df[col], percentiles[col])
        df["label"] = df["label"] + df[name + "_token"]

    return df


def downsample_label(df, label, n):
    rng = np.random.RandomState(0)
    return pd.concat(
        [df[df.label != label], df[df.label == label].sample(n, random_state=rng)],
        axis=1,
    ).copy()


def split_label_proportional(df, n_val, n_test):
    rng = np.random.RandomState(0)
    label_prop = df.label.value_counts(normalize=True)
    val_sample = np.ceil(label_prop * n_val)
    test_sample = np.ceil(label_prop * n_test)

    train_df = df.copy()
    val_df = pd.concat(
        [
            train_df[train_df.label == label].sample(
                val_sample[label], random_state=rng
            )
            for label in label_prop.index
        ],
        axis=1,
    )
    train_df = train_df.loc[~train_df.index.isin(val_df), :]
    test_df = pd.concat(
        [
            train_df[train_df.label == label].sample(
                test_sample[label], random_state=rng
            )
            for label in label_prop.index
        ],
        axis=1,
    )
    train_df = train_df.loc[~train_df.index.isin(test_df), :]
    return train_df, val_df, test_df


def split_label_equal(df, n_val, n_test):
    labels = np.unique(df.label)
    rng = np.random.RandomState(0)

    val_sample = np.ceil(n_val / len(labels))
    test_sample = np.ceil(n_test / len(labels))

    train_df = df.copy()
    val_df = pd.concat(
        [
            train_df[train_df.label == label].sample(val_sample, random_state=rng)
            for label in labels
        ],
        axis=1,
    )
    train_df = train_df.loc[~train_df.index.isin(val_df), :]
    test_df = pd.concat(
        [
            train_df[train_df.label == label].sample(test_sample, random_state=rng)
            for label in labels
        ],
        axis=1,
    )
    train_df = train_df.loc[~train_df.index.isin(test_df), :]
    return train_df, val_df, test_df


def seqs_to_idxs(seqs):
    base_to_idx = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }
    return np.array([[base_to_idx[base] for base in seq] for seq in seqs])


def scores_to_matrix(scores, seqs):
    """
    Convert per-base scores to a N x seq_len x 4 numpy array
    """
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
    Convert a 2D tensor of shape N x 4 to a 1-D array of shape N containing
    scores for the actual base
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
