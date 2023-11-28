import numpy as np
import pandas as pd

from reglm.utils import get_labels, get_percentiles, tokenize


def test_get_percentiles():
    assert np.all(
        get_percentiles([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], n_bins=4)
        == [2.75, 5.5, 8.25]
    )
    assert np.all(
        get_percentiles([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], qlist=[75]) == [8.25]
    )


def test_get_labels():
    assert np.all(
        get_labels(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], percentiles=[2.75, 5.5, 8.25]
        )
        == ["0", "0", "0", "1", "1", "1", "2", "2", "2", "3", "3", "3"]
    )


def test_tokenize():
    df = pd.DataFrame(
        {
            "C1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "C2": [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        }
    )
    assert np.all(
        tokenize(df, cols=["C1", "C2"], names=["C1", "C2"], n_bins=4)["label"].tolist()
        == ["03", "03", "03", "12", "12", "12", "21", "21", "21", "30", "30", "30"]
    )
    df = pd.DataFrame(
        {
            "C1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "C2": [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        }
    )
    assert np.all(
        tokenize(
            df,
            cols=["C1", "C2"],
            names=["C1", "C2"],
            percentiles={"C1": [8.25, 11.0], "C2": [8.25, 11.0]},
            n_bins=False,
        )["label"].tolist()
        == ["02", "01", "01", "00", "00", "00", "00", "00", "00", "10", "10", "20"]
    )
