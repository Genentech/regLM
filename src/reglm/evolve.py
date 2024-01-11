import numpy as np
import pandas as pd

from reglm.interpret import ISM
from reglm.regression import SeqDataset


def evolve(
    start_seqs,
    regression_model,
    seq_len=None,
    language_model=None,
    label=None,
    tol=0.0,
    specific=None,
    max_iter=10,
    device=0,
    num_workers=1,
    batch_size=512,
):
    """
    Directed evolution optionally using a language model to filter sequences.

    Args:
        start_seqs (list): Starting sequences
        regression_model (pl.LightningModule): Regression model
        seq_len (int): Sequence length for regression model
        language_model (pl.LightningModule): Language model
        label (str): Label for language model
        tol (float): Tolerance for likelihood filter
        specific (list): Task indices if optimizing for task specificity
        max_iter (int): Maximum number of iterations for evolution
        device (int): GPU index
        num_workers (int): Number of workers for regression model
        batch_size (int): Batch size for regression model

    Returns:
        df (pd.DataFrame): Dataframe containing evolution results
    """
    # Create empty dataframe for results
    df = pd.DataFrame()

    # Iterate
    for i in range(max_iter + 1):
        if i == 0:
            # initial dataframe includes only starting sequences
            curr_df = pd.DataFrame(
                {
                    "Sequence": start_seqs,
                    "iter": i,
                    "start_seq": range(len(start_seqs)),
                    "best_in_iter": [True] * len(start_seqs),
                }
            )

        elif i > 0:
            print(f"Iteration: {i}")
            start_seq_lens = [len(seq) for seq in start_seqs]

            # ISM
            new_seqs = np.concatenate([ISM(seq, drop_ref=True) for seq in start_seqs])
            curr_df = pd.DataFrame(
                {
                    "Sequence": new_seqs,
                    "start_seq": np.concatenate(
                        [
                            [s_idx] * s_len * 3
                            for s_idx, s_len in enumerate(start_seq_lens)
                        ]
                    ),
                    "iter": i,
                }
            )

        if language_model is not None:
            # Calculate likelihood
            curr_df["likelihood"] = np.concatenate(
                [
                    language_model.P_seqs_given_labels(
                        batch,
                        [label] * len(batch),
                        add_stop=True,
                        log=True,
                        per_pos=False,
                    )
                    for batch in np.split(
                        curr_df.Sequence.tolist(), list(range(1000, len(curr_df), 1000))
                    )
                ]
            )

            # Filter sequences based on whether the likelihood has improved
            # relative to their previous sequence
            if i > 0:
                # Get likelihood of the respective start sequence
                curr_df["prev_likelihood"] = curr_df.start_seq.apply(
                    lambda x: start_likelihoods[x]
                )
                # Filter
                curr_df = curr_df[curr_df.likelihood > (curr_df.prev_likelihood - tol)]

        # Predict function with regression model
        ds = SeqDataset(curr_df.Sequence.tolist())
        preds = regression_model.predict_on_dataset(
            ds, batch_size=batch_size, device=device, num_workers=num_workers
        )

        # Get mean prediction or task specificity
        if (specific is None) and (preds.ndim == 2):
            preds = preds.mean(1)
        else:
            non_specific = [x for x in range(preds.shape[1]) if x != specific]
            preds = preds[:, non_specific].max(1) - preds[:, specific]
        curr_df["pred"] = [x for x in preds]

        if i > 0:
            # Get the best sequence from each starting sequence
            curr_df["best_in_iter"] = [False] * len(curr_df)
            curr_df.loc[
                curr_df.groupby("start_seq").pred.idxmax(), "best_in_iter"
            ] = True

        # Collect the sequences to start the next iteration
        # And compute their likelihood
        start_seqs = curr_df.loc[curr_df.best_in_iter, "Sequence"].tolist()
        if language_model is not None:
            start_likelihoods = curr_df.loc[curr_df.best_in_iter, "likelihood"].tolist()

        # Concat to results
        df = pd.concat([df, curr_df])

    return df.reset_index(drop=True)
