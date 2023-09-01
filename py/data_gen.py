# %%
import argparse
from typing import Tuple
import numpy as np
from py.decisions import ThetaGenerator, Simulator


# for notebook.
def generate_data(
    num_applicants: int,
    applicants_per_round: int,
    fixed_effort_conversion: bool,
    args: argparse.Namespace,
    _theta: np.ndarray = None,
    _theta_star=None,
    fixed_competitors=False,
):
    """generates data by first generating theta, theta_star and then running the simulator.

    Args:
        num_applicants (int): number of applicants
        applicants_per_round (int): number of applicants in each round
        fixed_effort_conversion (bool): whether to use the same effort conversion
        matrix for each student.
        args (argparse.Namespace): namespace object from algos.py
        _theta (np.ndarray, optional): (T, n, m) matrix. deployment rules to use for each student.
        If given, uses that instead of generating
        a new one. Defaults to None.
        _theta_star (np.ndarray, optional): (n, m) matrix. ground truth theta for each university.
        If given, uses that instead of generating a new
        one. Defaults to None.
        fixed_competitors (bool, optional): whether to fix the competiting environments
        to use the same deployment rule. Used for
        computing theta_ao. Defaults to False.

    Returns (all following variables are of type np.ndarray):
        b_tr: (T, m) matrix. Baseline vector of students.
        x_tr: (T, m) matrix. Improved vector of students. 
        y: (n, m) matrix. College GPA of each student for each environment. 
        eet_mean: (m, m) matrix. mean of effort conversion matrix,
        theta: (n, T, m) matrix. deployed selection parameters
        w: (n, T) matrix. w[i, j] denotes with student j got an offer from college i.
        z: (T, ) vector. 
        y_hat: (n, T) predicted college GPA of each student for each uni.
        adv_idx: 1D vector. Contains indices of advantaged students. 
        disadv_idx, 1D vector. Contains indices of disadvantaged students. 
        o: (n, T) matrix. Exogeneous noise. 
        theta_star: (n, m) matrix. ground truth theta for each uni.
    """
    # pt. 1. ground truth causal parameters.
    if _theta_star is None:  # distribute randomly
        theta_star = np.zeros(shape=(args.num_envs, 2))
        theta_star[:, 1] = np.random.normal(
            loc=0.5, scale=args.theta_star_std, size=(args.num_envs,)
        )
    else:  # set as given
        theta_star = _theta_star

    assert num_applicants % applicants_per_round == 0
    n_rounds = int(num_applicants / applicants_per_round)

    # pt. 2. assessment rule
    if _theta is None:  # distribute randomly.
        thegen = ThetaGenerator(length=n_rounds, num_principals=args.num_envs)
        if args.scaled_duplicates is None:
            theta = thegen.generate_randomly()  # (T,n,m)
        elif args.scaled_duplicates == "sequence":
            theta = thegen.generate_general_coop_case(
                num_cooperative_principals=args.num_cooperative_envs
            )
        else:
            raise ValueError(args.scaled_duplicates)
    else:  # set as given
        assert _theta.shape == (args.num_envs, 2)  # (n,m)
        theta = np.tile(_theta, reps=(n_rounds, 1, 1))  # (T,n,m)

    # pt. 3. optionally fix all but the first principal.
    if fixed_competitors:
        # deployment rule of all but the first principal is fixed.
        for env_idx in range(1, args.num_envs):
            theta[:, env_idx, :] = theta[0, env_idx, :]

    theta, b_tr, x_tr, eet_mean, o, y, y_hat, w, z, adv_idx, disadv_idx = run_simulator(
        applicants_per_round, fixed_effort_conversion, args, theta_star, theta
    )

    return (
        b_tr,
        x_tr,
        y,
        eet_mean,
        theta,
        w,
        z,
        y_hat,
        adv_idx,
        disadv_idx,
        o.T,
        theta_star
    )


def run_simulator(
    applicants_per_round: int,
    fixed_effort_conversion: bool,
    args: argparse.Namespace,
    theta_star: np.ndarray,
    theta: np.ndarray,
):
    """Simulates generation and selection of students using deployment rules.

    Args:
        applicants_per_round (int): number of applicants
        fixed_effort_conversion (bool): whether to fix each student
        args (argparse.Namespace): namespace object from algos.py
        theta_star (np.ndarray): (n, m) matrix. ground truth causal coefficient of
        environments.
        theta (np.ndarray): (n, T, m) matrix. deployment rules of environments.
    """
    sim = Simulator(
        num_agents=applicants_per_round,
        has_same_effort=fixed_effort_conversion,
        does_clip=args.clip,
        does_normalise=args.normalize,
        ranking_type=args.rank_type,
    )
    sim.deploy(
        thetas_tr=theta, gammas=args.pref_vect, admission_rates=args.envs_accept_rates
    )
    u, b_tr, theta, x_tr, eet_mean = (
        sim.u,
        sim.b_tr,
        sim.thetas_tr,
        sim.x_tr,
        sim.eet_mean,
    )

    # true outcomes (college gpa)
    sim.enroll(theta_stars_tr=theta_star)
    o, y = sim.o, sim.y

    # for backwards compatibility
    theta = theta.transpose((1, 0, 2))
    y = y.T

    assert x_tr[np.newaxis].shape == (1, args.num_applicants, 2)
    assert theta.shape == (args.num_envs, args.num_applicants, 2)
    assert o.shape == (args.num_applicants, args.num_envs)
    assert theta_star.shape == (args.num_envs, 2)

    # our setup addition
    # computing admission results.
    y_hat = sim.y_hat.T
    w, z = sim.w_tr.T, sim.z

    # for backwards compatibility
    adv_idx = np.where(u == True)
    disadv_idx = np.where(u == False)
    adv_idx, disadv_idx = adv_idx[0], disadv_idx[0]
    return theta, b_tr, x_tr, eet_mean, o, y, y_hat, w, z, adv_idx, disadv_idx
