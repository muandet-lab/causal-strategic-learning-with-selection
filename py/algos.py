# %%
import argparse
import subprocess
from copy import deepcopy
import py.data_gen as data_gen

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from py.utils import recover_thetas
from py.data_gen import run_simulator


def get_git_revision_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


# %%
def get_args(cmd):
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--num-applicants", default=10000, type=int)
    parser.add_argument(
        "--applicants-per-round", default=1, type=int, help="used for identical thetas"
    )
    parser.add_argument("--fixed-effort-conversion", action="store_true")
    parser.add_argument(
        "--scaled-duplicates",
        default=None,
        choices=["sequence", None],
        type=str,
        help="whether to deploy scaled duplicates for each selection parameter "
        "(i.e. our settings) or not (original harris et. al settings).",
    )
    parser.add_argument(
        "--num-cooperative-envs",
        default=None,
        type=int,
        help="number of environments with shared frequency of how often to "
        "deploy scaled duplicates",
    )
    parser.add_argument(
        "--clip", action="store_true", help="clip, as in Harris. et al settings."
    )
    parser.add_argument(
        "--alpha",
        default=0,
        type=float,
        help="convex combination for deciding clipping thres.",
    )
    parser.add_argument(
        "--alpha-effort",
        default=1,
        type=float,
        help="convex combination for deciding mean and variance of added noise effort conversion matrix.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalise features to retain real-world interpretation",
    )
    parser.add_argument(
        "--theta-star-std",
        type=float,
        default=0,
        help="standard deviation of causal coefficient. 0 ensures identical causal coefficients of each env",
    )
    parser.add_argument(
        "--rank-type", type=str, default="prediction", choices=("prediction", "uniform")
    )

    # multienv
    parser.add_argument("--num-envs", default=1, type=int)
    parser.add_argument(
        "--envs-accept-rates",
        nargs="+",
        default=[1.00],
        type=float,
        help="acceptance rate of each env.",
    )
    parser.add_argument("--pref-vect", nargs="+", default=[1.00], type=float)

    # algorithm
    parser.add_argument(
        "--methods", choices=("ols", "2sls", "ours"), nargs="+", default="ours"
    )

    # misc
    parser.add_argument(
        "--offline-eval",
        action="store_true",
        help="evaluate " "the algorithms only once for the entire dataset together.",
    )

    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd.split(" "))

    if len(args.envs_accept_rates) == 1:
        args.envs_accept_rates = [args.envs_accept_rates[0]] * args.num_envs
    if len(args.pref_vect) == 1:
        args.pref_vect = [args.pref_vect[0]] * args.num_envs
    args.pref_vect = [p / sum(args.pref_vect) for p in args.pref_vect]
    args.pref_vect = np.array(args.pref_vect)

    if args.num_cooperative_envs is None:
        args.num_cooperative_envs = args.num_envs  # by default, everyone is cooperative

    assert len(args.envs_accept_rates) == args.num_envs
    assert len(args.pref_vect) == args.num_envs
    return args


eqs_data = {}
for i in (1, 0):
    for k in ("y", "b", "theta", "o", "theta_hat", "b_mean", "o_mean"):
        eqs_data[f"grp{i}_{k}"] = []
    # %%


# %%
def ols(x: np.ndarray, y: np.ndarray):
    """Applies OLS for causal param. estimation.

    Args:
        x (np.ndarray): (T, m) matrix of covariates
        y (np.ndarray): (*,) dimensional vector containing college GPAs
        of students enrolled in an env.

    Returns:
        theta_hat_ols (np.ndarray): (m,) dimensional vector. Estimated
        causal param. by OLS.
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    theta_hat_ols_ = model.coef_
    return theta_hat_ols_


def tsls(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray, pref_vect: np.ndarray
):  # runs until round T
    """Applies 2SLS method (Harris et. al) for causal parameter estimation.

    Args:
        x (np.ndarray): (T, m) matrix of covariates.
        y (np.ndarray): (*,) dimensional vector containing college GPAs of
        students enrolled in an env.
        theta (np.ndarray): (T, m) matrix. deployed selection parameters.
        pref_vect (np.ndarray): (n) dimensional vector. Preference vector.

    Returns:
        theta_hat_tsls (np.ndarray): estimated causal parameter by 2SLS algorithm.
    """
    # regress x onto theta: estimate omega
    model = LinearRegression()

    _pref_vect = pref_vect.reshape((pref_vect.shape[0], 1, 1))
    theta = (theta * _pref_vect).sum(axis=0)
    assert theta.shape == (y.shape[0], 2)

    model.fit(theta, x)
    omega_hat_ = model.coef_.T

    # regress y onto theta: estimate lambda
    model = LinearRegression()
    model.fit(theta, y)
    lambda_hat_ = model.coef_

    # estimate theta^*
    theta_hat_tsls_ = np.matmul(np.linalg.inv(omega_hat_), lambda_hat_)
    return theta_hat_tsls_


def our(x: np.ndarray, y: np.ndarray, theta: np.ndarray, w: np.ndarray):
    """Applies our proposed method for causal parameter estimation.

    Args:
        x (np.ndarray): (T, m) matrix of covariates.
        y (np.ndarray): (*,) dimensional vector containing college GPAs of
        students enrolled in an env.
        theta (np.ndarray): (T, m) matrix. deployed selection parameters.
        w (np.ndarray): (T,) dimensional binary vector. Denotes whether a
        corresponding student is enrolled at the college or not.

    Returns:
        theta_star_est (np.ndarray): (m,) dimensional vector. Estimated
        causal parameter.
    """
    model = LinearRegression()
    model.fit(theta, x)

    # recover scaled duplicate
    theta_admit = theta[w == 1]
    x_admit = x[w == 1]
    theta_admit_unique, counts = np.unique(theta_admit, axis=0, return_counts=True)

    # to use thetas having the most number of applicants
    idx = np.argsort(counts)
    idx = idx[::-1]
    theta_admit_unique = theta_admit_unique[idx]

    # construct linear system of eqs.
    assert theta_admit.shape[1] > 1  # algo not applicable for only one feature.
    n_eqs_required = theta_admit.shape[1]

    curr_n_eqs = 0
    A, b = [], []

    i = 0
    while i < theta_admit_unique.shape[0]:
        j = i + 1
        found_pair = False
        while (j < theta_admit_unique.shape[0]) and (found_pair == False):
            pair = theta_admit_unique[[i, j]]

            if (
                np.linalg.matrix_rank(pair) == 1
            ):  # if scalar multiple, and exact duplicates are ruled out.
                found_pair = True
                idx_grp1 = np.all(theta_admit == pair[1], axis=-1)
                idx_grp0 = np.all(theta_admit == pair[0], axis=-1)
                est1 = y[idx_grp1].mean()
                est0 = y[idx_grp0].mean()

                A.append(
                    (x_admit[idx_grp1].mean(axis=0) - x_admit[idx_grp0].mean(axis=0))[
                        np.newaxis
                    ]
                )
                b.append(np.array([est1 - est0]))

                curr_n_eqs += 1
            j += 1
        i += 1

    if curr_n_eqs < n_eqs_required:
        out = np.empty(shape=(n_eqs_required,))
        out[:] = np.nan
        return out

    A = np.concatenate(A, axis=0)
    b = np.concatenate(b)

    m = LinearRegression()
    m.fit(A, b)
    theta_star_est = m.coef_
    return theta_star_est


def run_multi_env_utility(args: argparse.Namespace, seed: int, theta_types: list):
    """calls run_multi_env() function (i.e. generates data, and estimates causal
    parameters) and then deploys a theta, depending on theta_type, and computes the
    utility for each environment.

    Args:
        args (argparse.Namespace): namespace object for arguments.
        seed (int): for reproducibility.
        theta_types (list): list of strings containing theta_type.

    Returns:
        utilities (np.ndarray): (n,) dimensional vector. Contains utility of
        each environment.
        test_thetas (np.ndarray) (1, n, m) matrix. contains deployed thetas
        for each env.
    """
    err_list, est_list, _, data_dict = run_multi_env(
        seed, args, env_idx=None, fixed_competitors=True
    )
    y, z, theta, theta_star = (
        data_dict["y"],
        data_dict["z"],
        data_dict["theta"],
        data_dict["theta_star"],
    )

    # replacing the estimate of causal parameter of fixed competitors with the ground truth one.
    for env_idx in range(1, args.num_envs):
        est_list[f"ours_env{env_idx}"] = [theta_star[env_idx]]
        err_list[f"ours_env{env_idx}"] = 0

    # recovering test thetas.
    test_thetas = np.zeros(shape=(args.num_envs, 2))
    for env_idx in range(args.num_envs):
        theta_type = theta_types[env_idx]
        test_thetas[env_idx] = recover_thetas(
            args.num_applicants,
            args.applicants_per_round,
            y,
            theta,
            z,
            env_idx,
            est_list,
            theta_type,
        )
    test_thetas = test_thetas[np.newaxis]

    # generating test data.
    args_test = deepcopy(args)
    args_test.num_applicants = args.applicants_per_round  # only one round
    _, _, _, _, _, y_test, _, _, z_test, _, _ = run_simulator(
        args_test.applicants_per_round,
        args_test.fixed_effort_conversion,
        args_test,
        theta_star,
        test_thetas,
    )
    y_test = y_test.T

    # compute utilities
    utilities = np.zeros((args_test.num_envs,))
    for env_idx in range(args_test.num_envs):
        utilities[env_idx] = y_test[:, env_idx][z_test == (env_idx + 1)].mean()

    return utilities, test_thetas


def run_multi_env(
    seed: int,
    args: argparse.Namespace,
    env_idx: int = None,
    fixed_competitors: bool = False,
):
    """Generates a dataset, and run causal parameter estimation methods.

    Args:
        seed (int): seed to ensure reproducibility.
        args (argparse.Namespace): arguments object.
        env_idx (int, optional): If given, run the algorithm for only env_idx env.
                                Defaults to None.
        fixed_competitors (bool, optional): Whether to fix deployment
                                             rules of other environments to a
                                             fixed value. Required for estimating
        theta_ao for utility maximisation. Defaults to False.

    Returns:
        err_list (dict): key is method x env_idx. Value is a list containing
                         estimation error over iterations.
        est_list (dict): key is (method x env_idx). Value is a list containing
                         estimated causal parameters over iterations.
        z (np.ndarray). (T,) vector. compliance of each student.
        data_dict (dict). Keys are ['theta_star', 'theta', 'z', 'y'].
    """
    np.random.seed(seed)
    _, x, y, _, theta, _, z, _, _, _, _, theta_star = data_gen.generate_data(
        args.num_applicants,
        args.applicants_per_round,
        args.fixed_effort_conversion,
        args,
        fixed_competitors=fixed_competitors,
    )
    data_dict = {"theta_star": theta_star, "theta": theta, "z": z, "y": y}

    err_list, est_list = {}, {}
    envs_itr = [env_idx] if env_idx is not None else range(args.num_envs)
    for env_idx in envs_itr:
        dictenv_err, dictenv_est = run_single_env(
            args, x, y, theta, z, theta_star, env_idx, args.pref_vect
        )
        for k, v in dictenv_err.items():
            err_list[f"{k}_env{env_idx}"] = v
        for k, v in dictenv_est.items():
            est_list[f"{k}_env{env_idx}"] = v

    return err_list, est_list, z, data_dict


def run_single_env(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    z: np.ndarray,
    theta_star: np.ndarray,
    env_idx: int,
    pref_vect: np.ndarray,
):
    """Runs each method for causal parameter estimation for a particular environment.

    Args:
        args (argparse.Namespace): namespace object containing arguments.
        x (np.ndarray): (T, m) matrix. covariates of students.
        y (np.ndarray): (n, T) matrix. college GPA of students.
        theta (np.ndarray): (n, T, m) matrix. deployed selection parameters.
        z (np.ndarray): (T,) dimensional vector. Contains compliance of each student.
        theta_star (np.ndarray): (n, m) matrix. ground truth causal coefficients.
        env_idx (int): index of environment for which to estimate causal param..
        pref_vect (np.ndarray): (n,) dimensional vector of preference.

    Returns:
        err_list (dict): key is method name. value is a list containing causal
                        estimation errors over iterations.
        est_list (dict): key is method name. value is a list containing estimated
                        casual coefficients over iterations.
    """
    # extracting relevant variables for the environment i.
    y_env = y[env_idx].flatten()
    theta_env = theta[env_idx]
    z_env = z == env_idx + 1

    upp_limits = [
        x
        for x in range(
            args.applicants_per_round * 2,
            args.num_applicants + 1,
            args.applicants_per_round,
        )
    ]
    if args.offline_eval:
        upp_limits = [args.num_applicants]

    err_list = {m: [None] * len(upp_limits) for m in args.methods}
    est_list = {m: [None] * len(upp_limits) for m in args.methods}
    for i, t in tqdm(enumerate(upp_limits)):
        x_round = x[:t]
        y_env_round = y_env[:t]
        z_env_round = z_env[:t]
        theta_env_round = theta_env[:t]

        # filtering out rejected students
        y_env_round_selected = y_env_round[z_env_round]
        x_round_selected = x_round[z_env_round]
        theta_round_selected = theta[:, :t][:, z_env_round]
        assert theta_round_selected.shape == (args.num_envs, z_env_round.sum(), 2)

        for m in args.methods:
            if m == "ours":
                est = our(x_round, y_env_round_selected, theta_env_round, z_env_round)
            elif m == "2sls":
                try:
                    est = tsls(
                        x_round_selected,
                        y_env_round_selected,
                        theta_round_selected,
                        pref_vect,
                    )
                except np.linalg.LinAlgError:
                    est = np.array([np.nan, np.nan])
            elif m == "ols":
                est = ols(x_round_selected, y_env_round_selected)

            assert (
                theta_star[env_idx].shape == est.shape
            ), f"{theta[0].shape}, {est.shape}"
            err_list[m][i] = np.linalg.norm(theta_star[env_idx] - est)
            est_list[m][i] = est

    return err_list, est_list


# convert to dataframe.
def runs2df(runs: list):
    """Converts a list of dictionaries to a dataframe.

    Args:
        runs (list): each entry is a dictionary. The dictionaries
        have identical keys, due to which it can be transformed into
        a pandas DataFrame.

    Returns:
        df (pd.DataFrame): Pandas Dataframe.
    """
    dfs = []
    for r in runs:
        df = pd.DataFrame(r)
        dfs.append(df)
    df = pd.concat(dfs)
    df.reset_index(inplace=True)
    df.rename({"index": "iterations"}, axis=1, inplace=True)
    return df
