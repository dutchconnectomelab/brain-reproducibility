import argparse
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io
import seaborn as sns
from alive_progress import alive_it
from statsmodels import stats

from brain_reproducibility import utils


def load_data(
    disorder: str,
    datapath=None,
    age_filter=None,
    rm_confounders: bool = True,
    atlas: str = "DK-114",
) -> Tuple:
    """Load data for a specific disorder and atlas.

    The data used is provided by a range of institutions and consortia (see README),
    and cannot be publicly distributed.
    Preprocessed data is available upon request to researchers who have signed the relevant Data Use Agreements.

    Args:
      datapath: path to directory on local system on which the data is stored.
      disorder: choice of disorder [schizophrenia, alzheimer]
      age_filter: optional age filter of the form [min_age, max_age]
      rm_confounders: whether the confounders of sex, site and age should be removed (if true, returned regionProperties are residuals)
      atlas: choice of atlas (used to select file) [DK-68, DK-114, DK-219, DK-448]

    Returns:
      pandas dataframe with participant info (diagnosis `dx`, `study`, and `cohort`), np array with cortical thickness values and np array with ROI names (only for empirical data, set to `None` for simulated data)
    """
    if datapath is None:
        print("No datapath provided; falling back to simulated data.")
        participants, ct = utils.simulate_data(atlas, disorder)
        return participants, ct, None

    sourcefile = Path(datapath) / f"cortical_thickness_{atlas}.mat"

    if not sourcefile.exists():
        raise RuntimeError(
            f"Could not find empirical data file {sourcefile}; please make sure the correct datapath is provided."
        )

    participants, ct, regions = utils.extract_data(scipy.io.loadmat(sourcefile))

    # filter subjects by disorder
    selection = [disorder in dx for dx in participants["dx"]]
    participants = participants[selection]
    ct = ct[:, selection]

    if len(participants) == 0:
        print(f"WARNING: no subjects for {disorder}")
        return participants, ct, disorder

    participants, ct = utils.remove_outliers(participants, ct)

    if age_filter is not None:
        participants, ct = utils.apply_age_filter(participants, ct, age_filter)

    if rm_confounders:
        ct = utils.regress_out_confounders(participants, ct)

    print(
        f"Including {len(participants)} subjects "
        f"({np.sum(participants['dx'] == disorder)} patients, "
        f"{np.sum(participants['dx'] == f'control_{disorder}')} controls) "
        f"for {disorder}"
    )

    return participants, ct, regions


def bootstrap_global(dx: np.ndarray, sample_size: int) -> dict:
    """Select disjoint sets of patients and controls for discovery and replication study

    Samples are drawn from the global population (i.e. each sample can contain subjects from different datasets.)

    Arguments:
      dx: numpy array with diagnosis of each subject (control dx should be "control")
      sample_size: number of subjects per sample

    Returns:
      dictionary with keys "disc_con", "disc_pat", "repl_con", "repl_pat", containing the IDs for each of these samples
    """

    candidates = np.arange(len(dx))
    controls = dx == "control"
    patients = ~controls

    selection_controls = np.random.choice(
        candidates[controls], 2 * sample_size, replace=False
    )
    selection_patients = np.random.choice(
        candidates[patients], 2 * sample_size, replace=False
    )

    return {
        "disc_pat": selection_patients[:sample_size],
        "disc_con": selection_controls[:sample_size],
        "repl_pat": selection_patients[sample_size:],
        "repl_con": selection_controls[sample_size:],
    }


def bootstrap_splitwise(dx, split, candidate_splits, sample_size) -> dict:
    """Select splits and sample patients and controls for discovery and replication study

    One split is selected for the discovery samples and a separate one for the replication samples.
    Thus subjects within a sample all come from the same split.

    Arguments:
      dx: numpy array with diagnosis of each subject ("control" for controls)
      split: array with split per subject
      candidate_splits: splits with enough patients and controls
      sample_size: number of subjects per sample

    Returns:
      dictionary with keys "disc_con", "disc_pat", "repl_con", "repl_pat", containing the IDs for each of these samples
    """
    candidates = np.arange(len(dx))

    split_selection = np.random.choice(candidate_splits, 2, replace=False)

    return {
        "disc_pat": np.random.choice(
            candidates[(dx != "control") & (split == split_selection[0])],
            sample_size,
            replace=False,
        ),
        "disc_con": np.random.choice(
            candidates[(dx == "control") & (split == split_selection[0])],
            sample_size,
            replace=False,
        ),
        "repl_pat": np.random.choice(
            candidates[(dx != "control") & (split == split_selection[1])],
            sample_size,
            replace=False,
        ),
        "repl_con": np.random.choice(
            candidates[(dx == "control") & (split == split_selection[1])],
            sample_size,
            replace=False,
        ),
    }


def case_control_study(patients, controls, thr):
    """Conduct case-control study

    Args:
      patients: np.array or pd.DataFrame with patient data (ROIs x SUBJ)
      controls: np.array or pd.DataFrame with control data (ROIs x SUBJ)
      thr: threshold type (either FDR or Bonferroni)

    Returns:
      Boolean array of significant regions
      Array with the direction of the effect
    """
    t, p = scipy.stats.ttest_ind(
        patients,
        controls,
        alternative="two-sided",
        axis=1,
    )

    if thr == "Bonferroni":
        significant = p < (0.05 / len(p))
    elif thr == "FDR":
        significant = stats.multitest.fdrcorrection(p)[1] < 0.05
    elif thr == "Uncorrected":
        significant = p < 0.05
    else:
        raise ValueError(f"Unknown threshold type {thr}")

    return significant, np.sign(t)


def estimate_replication_rate(
    disorder,
    data,
    sample_size,
    thr=0.05,
    iterations=1000,
    sampling_mode=0,
) -> pd.DataFrame:
    """Estimate replication rate for given settigs by running repeated discovery and replication studies"""

    participants, ct, _ = data

    # add simplified diagnosis label to facilitate bootstrapping
    participants["dx_plain"] = [
        "control" if "control" in dx else "patient" for dx in participants["dx"]
    ]

    if len(participants) == 0:
        return pd.DataFrame()

    control_dx = f"control_{disorder}"

    n_pat = (participants["dx"] == disorder).sum()
    n_ctrl = (participants["dx"] == control_dx).sum()

    if (n_pat < 2 * sample_size) | (n_ctrl < 2 * sample_size):
        print(
            f"Skipping sample size = {sample_size}, not enough subjects to conduct a discovery and replication study."
        )
        return pd.DataFrame()

    if sampling_mode > 0:
        # compute which datasets can be used for which sample sizes

        split = (
            participants["cohort"] if sampling_mode == 1 else participants["dataset"]
        )
        split_names = np.unique(split)
        split_max = {
            s: np.min(
                [
                    np.sum(participants["dx"][split == s] == control_dx),
                    np.sum(participants["dx"][split == s] == disorder),
                ]
            )
            for s in split_names
        }
        candidate_splits = [ds for ds in split_names if split_max[ds] >= sample_size]

        if len(candidate_splits) < 2:
            print(
                f"Skipping sample size = {sample_size}, need at least splits with enough subjects."
            )
            return pd.DataFrame()

    replication_rate = np.zeros((iterations))

    iter = 0
    while iter < iterations:

        if sampling_mode == 0:
            sample_ids = bootstrap_global(participants["dx_plain"], sample_size)
        else:
            sample_ids = bootstrap_splitwise(
                participants["dx_plain"], split, candidate_splits, sample_size
            )

        significant_disc, dir_disc = case_control_study(
            ct[:, sample_ids["disc_pat"]],
            ct[:, sample_ids["disc_con"]],
            thr,
        )

        if sum(significant_disc) == 0:
            continue

        significant_rep, dir_rep = case_control_study(
            ct[:, sample_ids["repl_pat"]],
            ct[:, sample_ids["repl_con"]],
            thr,
        )

        overlap = significant_disc & significant_rep & (dir_disc == dir_rep)

        replication_rate[iter] = np.sum(overlap) / np.sum(significant_disc)

        iter += 1

    return pd.DataFrame(
        {
            "sample_size": sample_size,
            "replication_rate": replication_rate,
            "thr": thr,
            "disorder": disorder,
        },
        index=np.arange(iterations),
    )


def generate_figure(res: pd.DataFrame) -> mpl.figure.Figure:
    """Generate figure (under default arguments corresponds to Figure 1b)"""
    sample_count = [25, 50, 100, 200, 400, 800]

    sns.set(font_scale=2.5)
    fig = plt.figure(1, [11, 7])
    res["rr_perc"] = res["replication_rate"] * 100
    g = sns.lineplot(
        x="sample_size",
        y="rr_perc",
        linestyle="-",
        hue="disorder",
        data=res,
        linewidth=5,
        ci="sd",
    )

    g.set(xscale="log")
    g.set(xticks=sample_count)
    g.set(xticklabels=sample_count)
    g.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    plt.xlim([25, 1100])
    plt.xlabel("Sample size")
    plt.ylabel("Successful replication (%)")
    plt.ylim([-5, 105])
    plt.legend(title="", loc="upper left")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations",
        default=1000,
        type=int,
        help="Number of iterations for each significant level/sample size",
    )
    parser.add_argument(
        "-s",
        "--sampling-mode",
        default=0,
        type=int,
        help="Controls sampling procedure: (0) global sampling, i.e. across dataset and cohorts; (1) cohort sampling (see method section of paper); (2) within-dataset sampling",
    )
    parser.add_argument(
        "-d",
        "--datapath",
        type=str,
        help="Path on your local system to empirical data (see README for more info)",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default="DK-114",
        help="Atlas to be used [DK-68, DK-114, DK-219, DK-448]",
    )
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    args = parser.parse_args()

    # settings and data
    np.random.seed(0)
    disorders = ["alzheimer", "schizophrenia"]
    sample_sizes = {
        "alzheimer": [25, 33, 50, 70, 100, 135, 200, 265, 400, 525, 725, 1000, 1100],
        "schizophrenia": [25, 33, 45, 60, 80, 100, 145, 200, 230, 256, 350, 400, 430],
    }
    age_filter = {"alzheimer": [50, 99], "schizophrenia": [18, 60]}
    try:
        data = {
            disorder: load_data(
                disorder,
                args.datapath,
                age_filter=age_filter[disorder],
                rm_confounders=True,
                atlas=args.atlas,
            )
            for disorder in disorders
        }
    except RuntimeError as e:
        print(e)
        print(
            "If no empirical data is available, it is also possible to run the script without --datapath "
            "to use simulated data matching the mean and standard deviation of the empirical data."
        )
        exit()

    # run bootstrap simulations and collect results
    res = pd.concat(
        [
            estimate_replication_rate(
                disorder=disorder,
                data=data[disorder],
                thr="Bonferroni",
                sample_size=sample_size,
                iterations=args.iterations,
                sampling_mode=args.sampling_mode,
            )
            for disorder in disorders
            for sample_size in alive_it(
                sample_sizes[disorder], title=disorder, disable=not args.progress
            )
        ],
        ignore_index=True,
    )

    generate_figure(res)
    plt.show()
