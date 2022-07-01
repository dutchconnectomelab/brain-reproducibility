import numpy as np

from brain_reproducibility import replication


def check_global_sampling(samples, dx, n):
    # check sample sizes
    assert (
        len(samples["disc_con"])
        == len(samples["disc_pat"])
        == len(samples["repl_con"])
        == len(samples["repl_pat"])
        == n
    )

    # check duplicates
    assert (
        len(np.unique(samples["disc_con"]))
        == len(np.unique(samples["disc_pat"]))
        == len(np.unique(samples["repl_con"]))
        == len(np.unique(samples["repl_pat"]))
        == n
    )

    # check samples are of correct diagnosis and don't overlap
    for i in range(n):
        assert dx[samples["disc_con"][i]] == "control"
        assert dx[samples["repl_con"][i]] == "control"
        assert dx[samples["disc_pat"][i]] == "schizophrenia"
        assert dx[samples["repl_pat"][i]] == "schizophrenia"

        assert samples["disc_con"][i] not in samples["repl_con"]
        assert samples["disc_pat"][i] not in samples["repl_pat"]


def test_global_bootstrapping():
    dx = np.array(
        [
            {0: "control", 1: "schizophrenia"}[x]
            for x in np.random.randint(0, 2, size=(10000))
        ]
    )

    n = 100  # sample size

    samples = replication.bootstrap_global(dx, n)
    check_global_sampling(samples, dx, n)


def test_split_bootstrapping():
    dx = np.array(
        [
            {0: "control", 1: "schizophrenia"}[x]
            for x in np.random.randint(0, 2, size=(10000))
        ]
    )
    split = np.random.randint(0, 3, size=(10000))

    n = 100  # sample size

    samples = replication.bootstrap_splitwise(dx, split, [0, 1, 2], n)

    # ensure splits are disjoint and valid
    check_global_sampling(samples, dx, n)

    # ensure discovery samples come from the same split; same for replication
    assert (
        len(
            np.unique(split[np.concatenate([samples["disc_con"], samples["disc_pat"]])])
        )
        == 1
    )
    assert (
        len(
            np.unique(split[np.concatenate([samples["repl_con"], samples["repl_pat"]])])
        )
        == 1
    )

    # ensure discovery and replication splits are different
    assert split[samples["disc_con"][0]] != split[samples["repl_con"][0]]


def test_case_control_study():
    n_subj = 100
    n_roi = 500

    # test without any effect
    patients = np.random.randn(n_roi, n_subj)
    controls = np.random.randn(n_roi, n_subj)

    significant, dir = replication.case_control_study(patients, controls, "Uncorrected")

    assert 0.01 < np.mean(significant) < 0.1
    assert np.abs(np.mean(dir)) < 0.2

    significant, _ = replication.case_control_study(patients, controls, "Bonferroni")

    assert np.sum(significant) < 5

    # test for very large effect
    patients = np.random.randn(n_roi, n_subj) + 100
    controls = np.random.randn(n_roi, n_subj)

    significant, dir = replication.case_control_study(patients, controls, "Bonferroni")

    assert np.mean(significant) > 0.9
    assert np.mean(dir) > 0.9
