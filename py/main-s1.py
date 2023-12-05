from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from py.agents_gen import DEFAULT_AGENTS_MODEL
from py.decisions import Simulator

OUT_DIR = "csl-code"


class ExpName(str, Enum):
    GPA01 = 'HSGPA-01'
    GPA02 = 'HSGPA-02'
    SAT01 = 'SAT-01'
    SAT02 = 'SAT-02'


def run_S1_exp(name: ExpName):
    # params
    T = 600
    s = 200
    gammas = [1]
    admission_rates = [0.6]
    theta_stars_tr = np.array([[0.0, 0.5]]) # [SAT, HS GPA]
    # theta_stars_tr = np.array([[0.5, 0.0]])

    # customise my agents' model
    am = DEFAULT_AGENTS_MODEL

    # generate thetas, (T,1,2)
    thetas_tr = {
        ExpName.GPA01: np.random.uniform(low=[1, -40], high=[1, 40], size=[T, 1, 2]),
        ExpName.GPA02: np.random.uniform(low=[-1, -40], high=[-1, 40], size=[T, 1, 2]),
        ExpName.SAT01: np.random.uniform(low=[-40, 1], high=[40, 1], size=[T, 1, 2]),
        ExpName.SAT02: np.random.uniform(low=[-40, -1], high=[40, -1], size=[T, 1, 2])
    }[name]

    # deploy
    sim = Simulator(
        num_agents=s,
        has_same_effort=True,
        does_clip=False,
        does_normalise=False,
        ranking_type="prediction",
        agents_model=am,
    )
    sim.deploy(thetas_tr=thetas_tr, gammas=gammas, admission_rates=admission_rates)

    # compute cBP by assuming x=b
    _, y = am.gen_outcomes(u=sim.u, x_tr=sim.b_tr, theta_stars_tr=theta_stars_tr)
    w = sim.w_tr.T[0, :]

    y = y[:, 0]  # extract the outcomes in env 1
    cBP = [
        y[i * s: (i + 1) * s][w[i * s: (i + 1) * s] == 1].mean() for i in range(T)
    ]  # compute the mean in each round (i.e., from i*s to (i+1)*s)

    # plot cBP and thetas_tr[:,0,1] or thetas_tr[:,0,0]
    varying_thetas = {
        ExpName.GPA01: thetas_tr[:, 0, 1], ExpName.GPA02: thetas_tr[:, 0, 1],
        ExpName.SAT01: thetas_tr[:, 0, 0], ExpName.SAT02: thetas_tr[:, 0, 0]
    }[name]

    xlabel = {
        ExpName.GPA01: r'$\theta_t^{HS GPA}$', ExpName.GPA02: r'$\theta_t^{HS GPA}$',
        ExpName.SAT01: r'$\theta_t^{SAT}$', ExpName.SAT02: r'$\theta_t^{SAT}$'
    }[name]

    title = {
        ExpName.GPA01: r'When $\theta_t^{SAT}=1$', ExpName.GPA02: r'When $\theta_t^{SAT}=-1$',
        ExpName.SAT01: r'When $\theta_t^{HS GPA}=1$', ExpName.SAT02: r'When $\theta_t^{HS GPA}=-1$'
    }[name]

    fig = plt.figure(figsize=(5,3))
    plt.scatter(varying_thetas, cBP)
    plt.ylabel('cBP')
    plt.xlabel(xlabel)
    plt.title(title)
    fig.savefig(f'{OUT_DIR}/S1-cBP-theta-{name}.png', bbox_inches='tight')

    return


if __name__ == "__main__":
    for name in tqdm(ExpName):
        run_S1_exp(name)
    # run_S1_exp(ExpName.GPA01)
