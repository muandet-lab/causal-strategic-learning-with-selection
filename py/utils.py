from sklearn.linear_model import LinearRegression
from typing import List, Tuple

import numpy as np


def normalize(arrs: list, new_min: float, new_max: float) -> Tuple[List[float], float]:
    new_range = new_max - new_min
    curr_min = np.concatenate(arrs).min()
    curr_max = np.concatenate(arrs).max()

    curr_range = curr_max - curr_min

    out = []
    for arr in arrs:
        out.append((((arr - curr_min) / curr_range) * new_range) + new_min)
    return out, new_range / curr_range


def recover_thetas(
    num_applicants: int,
    applicants_per_round: int,
    y: np.ndarray,
    theta: np.ndarray,
    z: np.ndarray,
    env_idx: int,
    est_list: dict,
    theta_type: str,
):
    """Having estimated theta_star, returns a theta, depending on theta_type, 
    for environment env_idx. 

    Args:
        num_applicants (int): number of applicants
        applicants_per_round (int): applicants per round
        y (np.ndarray): (n, T) matrix. 
        theta (np.ndarray): (n, T, m) matrix. 
        z (np.ndarray): (T,) dimensional vector. 
        env_idx (int): index of environment for which to recover the theta to deploy.
        est_list (dict): key is (method * env_idx). value is a list containing 
        estimated causal parameter over iterations. 
        theta_type (str): type of theta to deploy. 

    Returns:
        theta_star, theta_ao, or theta_ols (np.ndarray): (m,) dimensional vector. 
        The vector to deploy for future students. 
    """
    assert theta_type in ("theta_star_hat", "theta_ols_hat", "theta_ao_hat")

    # compute theta star norm
    theta_star_est = est_list[f"ours_env{env_idx}"][-1]
    theta_star_est_norm = np.linalg.norm(theta_star_est)

    if theta_type == "theta_star_hat":
        theta_star_est = est_list[f"ours_env{env_idx}"][-1]
        return theta_star_est

    elif theta_type == "theta_ols_hat":
        theta_ols = est_list[f"ols_env{env_idx}"][-1]
        theta_ols_norm = np.linalg.norm(theta_ols)

        if (
            theta_ols_norm > theta_star_est_norm
        ):  # scale theta ols down, in case it has larger magnitude.
            theta_ols *= (
                theta_star_est_norm / theta_ols_norm
            )  # same magnitude as theta_star_hat
        return theta_ols

    elif theta_type == "theta_ao_hat":
        # recovering theta_ao
        theta_ao_target, theta_ao_input = [], []
        n_rounds = num_applicants / applicants_per_round

        for t in range(int(n_rounds)):
            lower = t * applicants_per_round
            upper = lower + applicants_per_round

            idx = z[lower:upper] == env_idx + 1
            theta_ao_target.append(y[env_idx, lower:upper][idx].mean())
            theta_ao_input.append(theta[env_idx, lower])

        theta_ao_input, theta_ao_target = np.array(theta_ao_input), np.array(
            theta_ao_target
        )
        m = LinearRegression()
        m.fit(theta_ao_input, theta_ao_target)
        theta_ao_est = m.coef_
        theta_ao_est *= theta_star_est_norm / np.linalg.norm(
            theta_ao_est
        )  # same magnitude as theta_star_hat
        return theta_ao_est
