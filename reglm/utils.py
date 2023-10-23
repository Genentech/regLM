import numpy as np


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
    Return labels for sequences
    """
    return [str(x) for x in np.digitize(values, percentiles)]


def tokenize(df, cols, names, n_bins, percentiles=None):
    if percentiles is None:
        percentiles = dict()
        for col in cols:
            percentiles[col] = get_percentiles(df[col], n_bins=n_bins)
            print(col, percentiles[col].tolist())

    for name, col in zip(names, cols):
        df[name + "_token"] = get_labels(df[col], percentiles[col])
        if "label" in df.columns:
            df["label"] = df["label"] + df[name + "_token"]
        else:
            df["label"] = df[name + "_token"]

    return df
