from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm


def cohen_d(x, y, axis=0):
    """compute Cohen's d"""
    return (np.nanmean(x, axis=axis) - np.nanmean(y, axis=axis)) / np.sqrt(
        (np.nanstd(x, ddof=1, axis=axis) ** 2 + np.nanstd(y, ddof=1, axis=axis) ** 2)
        / 2.0
    )


def resid_dist(dv, iv):
    """Compute residuals regressing out independient variable (iv) from dependent variable (dv)"""
    dv = dv.squeeze()
    cnd1 = ~np.isnan(dv)
    cnd2 = (~np.isnan(iv.astype(float))).all(axis=1)
    dv_fin = np.array(dv[cnd1 & cnd2])
    iv_fin = np.array(iv[cnd1 & cnd2].astype(float))

    x = np.asarray(sm.add_constant(iv_fin).astype(float))

    # fit linear regression model
    model = sm.OLS(dv_fin, x, missing="drop").fit()

    # create instance of influence
    influence = model.get_influence()

    # obtain standardized residuals
    standardized_residuals = influence.resid_studentized_internal

    return standardized_residuals


def unpack(x):
    """Unpack matlab data to pandas Series"""
    return pd.Series(np.dstack(np.concatenate(x)).flatten())


def extract_data(data):
    """Extract data"""
    dd = dict()
    dd["dataset"] = unpack(data["dataset"]).astype(str)
    dd["age"] = unpack(data["age"]).astype(float)
    dd["sex"] = unpack(data["sex"]).astype(str)
    dd["dx"] = unpack(data["dx"]).astype(str)
    dd["cohort"] = unpack(data["cohort"]).astype(str)

    ct = data["regionProperties"]
    regions = unpack(data["regionDescriptions"]).astype(str)

    # check dimensions
    assert ct.ndim == 2
    assert (
        dd["dataset"].shape[0]
        == dd["age"].shape[0]
        == dd["sex"].shape[0]
        == dd["dx"].shape[0]
        == dd["cohort"].shape[-1]
        == ct.shape[-1]
    )

    return pd.DataFrame(dd), ct, regions


def simulate_data(atlas: str, disorder: str):
    statdir = Path(__file__).parent / "assets" / "summarystats"
    population = pd.read_csv(popfile := statdir / f"{disorder}_{atlas}_population.csv")
    sumstats = pd.read_csv(statfile := statdir / f"{disorder}_{atlas}.csv")

    print(
        f"Simulating data for {population['count'].sum()} subjects based on the summary statistics "
        f"in {statfile} and population numbers in {popfile}..."
    )

    n_rois = len(np.unique(sumstats["roi"]))
    ct = []
    participants = []
    for _, r in population.iterrows():
        participants.append(
            pd.DataFrame(
                {"cohort": r["cohort"], "dataset": r["dataset"], "dx": r["dx"]},
                index=np.arange(r["count"]),
            )
        )
        study_stats = sumstats[
            (sumstats["dataset"] == r["dataset"]) & (sumstats["dx"] == r["dx"])
        ]
        assert len(study_stats) == n_rois
        std = np.expand_dims(study_stats["std"], -1)
        mean = np.expand_dims(study_stats["mean"], -1)
        ct.append(np.random.randn(n_rois, r["count"]) * std + mean)

    return pd.concat(participants), np.concatenate(ct, axis=-1)


def apply_age_filter(
    participants: pd.DataFrame, ct: np.ndarray, age_filter: Sequence[int]
):
    selection = [age_filter[0] <= age <= age_filter[1] for age in participants["age"]]
    return participants[selection], ct[:, selection]


def remove_outliers(participants: pd.DataFrame, ct: np.ndarray, z_thr=3):
    ct_mean = np.mean(ct, axis=0)
    z_stat = scipy.stats.zscore(ct_mean, nan_policy="omit", axis=0)
    selection = np.abs(z_stat) < z_thr
    n_outliers = len(selection) - np.sum(selection)
    if n_outliers > 0:
        print(f"Removing {n_outliers} outliers from data")
    return participants[selection], ct[:, selection]


def regress_out_confounders(participants: pd.DataFrame, ct: np.ndarray) -> np.ndarray:
    sex_dummy = pd.get_dummies(participants["sex"]).iloc[:, 0]
    site_dummy = pd.get_dummies(participants["dataset"])
    cov = pd.concat([sex_dummy, site_dummy, participants["age"].astype(float)], axis=1)
    for i in range(ct.shape[0]):
        ct[i, :] = resid_dist(ct[i, :], cov)
    return ct
