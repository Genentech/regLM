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


def downsample_label(df, label, n):
    rng = np.random.RandomState(0)
    return pd.concat([df[df.label!=label], df[df.label==label].sample(n, random_state=rng)], axis=1).copy()


def split_label_proportional(df, n_val, n_test):
    rng = np.random.RandomState(0)
    label_prop = df.label.value_counts(normalize=True)
    val_sample = np.ceil(label_prop * n_val)
    test_sample = np.ceil(label_prop * n_test)

    train_df = df.copy()
    val_df = pd.concat([train_df[train_df.label==label].sample(val_sample[label], random_state=rng) for label in prop.index], axis=1)
    train_df = train_df.loc[~train_df.index.isin(val_df), :]
    test_df = pd.concat([train_df[train_df.label==label].sample(test_sample[label], random_state=rng) for label in prop.index], axis=1)
    train_df = train_df.loc[~train_df.index.isin(test_df), :]
    return train_df, val_df, test_df


def split_label_equal(df, n_val, n_test):
    labels = np.unique(df.label)
    rng = np.random.RandomState(0)
    
    val_sample = np.ceil(n_val/len(labels))
    test_sample = np.ceil(n_test/len(labels))

    train_df = df.copy()
    val_df = pd.concat([train_df[train_df.label==label].sample(val_sample, random_state=rng) for label in labels], axis=1)
    train_df = train_df.loc[~train_df.index.isin(val_df), :]
    test_df = pd.concat([train_df[train_df.label==label].sample(test_sample, random_state=rng) for label in labels], axis=1)
    train_df = train_df.loc[~train_df.index.isin(test_df), :]
    return train_df, val_df, test_df
    