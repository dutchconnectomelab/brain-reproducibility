import argparse
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from alive_progress import alive_bar


def create_ground_truth(
    enigma_summary_statistics: dict, p_threshold: float
) -> np.ndarray:
    """Generate a ground truth array based on ENIGMA summary statistics"""
    cohen_d = np.array(enigma_summary_statistics["d_icv"].to_list())
    significant = enigma_summary_statistics["pobs"] < p_threshold

    # make sure we have both ground truth positives and ground truth negatives
    assert np.sum(significant) > 0
    assert np.sum(~significant) > 0

    ground_truth = cohen_d
    ground_truth[~significant] = 0

    return ground_truth


def simulate_study(
    sample_size: int,
    alpha: float,
    ground_truth_effect_size: np.ndarray,
) -> Sequence[float]:
    """Simulate an individual study and return observed power, false positive rate and false negative rate"""
    ground_truth = ground_truth_effect_size != 0
    n_rois = len(ground_truth)
    patients = np.random.randn(n_rois, sample_size) + np.expand_dims(
        ground_truth_effect_size, axis=1
    )
    controls = np.random.randn(n_rois, sample_size)

    observed_effect_size = np.mean(patients, axis=1) - np.mean(controls, axis=1)
    pvals = scipy.stats.ttest_ind(patients, controls, axis=1)[1]

    detected = pvals < alpha

    correctly_detected = np.bitwise_and(
        detected,
        np.sign(observed_effect_size) == np.sign(ground_truth_effect_size),
    )

    pwr = sum(correctly_detected[ground_truth]) / sum(ground_truth)
    fpr = sum(detected[~ground_truth]) / sum(~ground_truth)
    fnr = (sum(ground_truth) - sum(correctly_detected[ground_truth])) / sum(
        ground_truth
    )

    return pwr, fpr, fnr


def power_simulation(
    ground_truth,
    sample_size: int,
    alpha: float,
    iterations: int,
    n_rois: int,
) -> pd.DataFrame:
    """Run repeated study simulations and create dataframe with average results"""

    if len(ground_truth) != n_rois:
        ground_truth = np.concatenate(
            [ground_truth, np.random.choice(ground_truth, n_rois - len(ground_truth))]
        )

    mean_pwr, mean_fpr, mean_fnr = np.array(
        [
            simulate_study(
                sample_size,
                alpha=alpha,
                ground_truth_effect_size=ground_truth,
            )
            for _ in range(iterations)
        ]
    ).mean(axis=0)
    return pd.DataFrame(
        {
            "sample_size": sample_size,
            "power": mean_pwr,
            "false_positive_rate": mean_fpr,
            "false_negative_rate": mean_fnr,
            "alpha": alpha,
        },
        index=[0],
    )


def load_marek_reference_data() -> pd.DataFrame:
    marek_3a = pd.read_csv(
        Path(__file__).parent / "assets" / "marek_fig3a_annotated.csv"
    )
    marek_3a.columns = [
        "sample_size",
        "0.0000001",
        "0.000001",
        "0.00001",
        "0.0001",
        "0.001",
        "0.01",
        "0.05",
    ]
    marek_3a = marek_3a.melt("sample_size")
    marek_3a.columns = ["sample_size", "alpha", "false_negative_rate"]
    marek_3a["power"] = 100 - marek_3a["false_negative_rate"]
    marek_3a["power"] = marek_3a["power"] / 100
    marek_3a["category"] = "Marek et al."
    return marek_3a


def generate_figure(df: pd.DataFrame) -> mpl.figure.Figure:
    """Generate figure (under default arguments corresponds to Figure 1a)"""

    df["power_perc"] = df["power"] * 100
    df = df.replace(1e-05, "0.00001")
    df["P values"] = df["alpha"].astype(str)

    sns.set(font_scale=2.5)
    fig = plt.figure(1, [13, 7.5])

    sample_count = [25, 50, 100, 200, 400, 800, 1600, 3200]

    g = sns.lineplot(
        x="sample_size",
        y="power_perc",
        style="category",
        hue="P values",
        data=df,
        palette="mako_r",
        linewidth=5,
    )
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[-2:], labels[-2:])
    g.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    g.set(xscale="log")
    g.set(xticks=sample_count)
    g.set(xticklabels=sample_count)
    plt.xlabel("Sample size")
    plt.ylabel("Statistical power (%)")
    plt.axhline([80], color="black", linestyle="-.", alpha=0.6)
    plt.ylim([-5, 105])
    plt.xlim([25, 4000])
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
        "--rois",
        default=114,
        type=int,
        help="Number of regions of interests included in the simulations",
    )
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    args = parser.parse_args()

    np.random.seed(0)

    enigma_summary_statistics = pd.read_csv(
        Path(__file__).parent / "assets" / "ENIGMA_scz.csv"
    )

    pvals = (0.05, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    sample_sizes = (
        25,
        33,
        50,
        70,
        100,
        135,
        200,
        265,
        375,
        525,
        725,
        1000,
        1430,
        2000,
        2800,
        4000,
    )

    results = []
    with alive_bar(len(pvals) * len(sample_sizes), disable=not args.progress) as bar:
        for pval in pvals:
            ground_truth = create_ground_truth(
                enigma_summary_statistics, p_threshold=pval
            )
            for sample_size in sample_sizes:
                df_sim = power_simulation(
                    ground_truth,
                    sample_size,
                    pval,
                    iterations=args.iterations,
                    n_rois=args.rois,
                )
                df_sim["category"] = "Schizophrenia"
                results.append(df_sim)
                bar()

    results.append(load_marek_reference_data())

    df = pd.concat(results, ignore_index=True)

    generate_figure(df)

    plt.show()
