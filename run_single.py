"""
Experiment E0: one elevator (the starting point).

    python run_single.py          # about 5 minutes on 2 cores

Writes results/single/single_results.json, two figures and
results/single/single_elevator_results.pdf.

Question: for a single elevator, how far is the classical LOOK rule from
the truly optimal policy, and can Q-learning find that optimum from
experience alone? The model is small enough (1200 states) to solve
exactly with value iteration, so we know the real optimum.
"""

import json
import os
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elevator.single_elevator import (
    build_model, value_iteration, evaluate_policy, table_policy, look_action,
    q_learning, run_policy, long_run_cost, STATES, INDEX, ACTIONS, VALID, N_STATES)

OUT = "results/single"
N_TEST = 200            # test trials of 250 steps (500 seconds), as in the paper
TEST_SEED0 = 50_000
EPISODES = 30_000       # short exploring-start episodes
EPISODE_STEPS = 50
SEEDS = 3
GAMMAS = (0.95, 0.99)

P, R = build_model()


def test(policy):
    """Average metrics of a fixed policy over the same test trials."""
    runs = [run_policy(lambda s, sim: policy[s], TEST_SEED0 + i) for i in range(N_TEST)]
    return dict(people=float(np.mean([r["people"] for r in runs])),
                journey=float(np.nanmean([r["journey"] for r in runs])))


def train(job):
    gamma, seed = job
    t0 = time.time()
    curve = []
    checkpoints = set(np.linspace(0, EPISODES, 11).astype(int)[1:])

    def record(ep, pi):
        runs = [run_policy(lambda s, sim: pi[s], TEST_SEED0 + i) for i in range(50)]
        curve.append(dict(episode=ep, people=float(np.mean([r["people"] for r in runs]))))

    pi, _ = q_learning(EPISODES, EPISODE_STEPS, gamma, seed,
                       on_checkpoint=record, checkpoints=checkpoints)
    return dict(gamma=gamma, seed=seed, policy=pi.tolist(), curve=curve,
                test=test(pi), seconds=time.time() - t0)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = dict(settings=dict(n_test=N_TEST, episodes=EPISODES, episode_steps=EPISODE_STEPS,
                             seeds=SEEDS, gammas=GAMMAS))

    # --- model-based: exact solutions -----------------------------------
    look = table_policy(look_action)
    rng = np.random.default_rng(0)
    rand_runs = [run_policy(lambda s, sim: int(rng.choice(VALID[s])), TEST_SEED0 + i)
                 for i in range(N_TEST)]
    res["random"] = dict(people=float(np.mean([r["people"] for r in rand_runs])),
                         journey=float(np.nanmean([r["journey"] for r in rand_runs])))
    res["look"] = test(look)
    res["look"]["longrun"] = long_run_cost(P, look)
    res["optimal"] = {}
    empty_idle = [INDEX[(0, 0, 0, 0, p, 0, 0)] for p in range(5)]
    for g in GAMMAS:
        pi, Q, iters = value_iteration(P, R, g)
        res["optimal"][str(g)] = dict(
            test=test(pi), longrun=long_run_cost(P, pi), iterations=iters,
            # value of an empty building, from the MRP formula V = (I - gamma P_pi)^-1 R_pi
            v_empty=float(evaluate_policy(P, R, pi, g)[empty_idle[0]]),
            v_empty_look=float(evaluate_policy(P, R, look, g)[empty_idle[0]]),
            idle_moves={p: ACTIONS[pi[i]] for p, i in enumerate(empty_idle)},
            policy=pi.tolist())
        print(f"value iteration gamma={g}: {iters} iterations, "
              f"{res['optimal'][str(g)]['test']['people']:.3f} people in system")

    # how often LOOK does the same as the optimal policy, weighted by how often
    # the optimal policy visits each state
    pi99 = np.array(res["optimal"]["0.99"]["policy"])
    idx = np.arange(N_STATES)
    P_pi = P[idx, pi99]
    dist, occ = np.zeros(N_STATES), np.zeros(N_STATES)
    dist[INDEX[(0, 0, 0, 0, 0, 0, 0)]] = 1.0
    for _ in range(5000):
        dist = dist @ P_pi
        occ += dist
    occ /= occ.sum()
    res["look_agreement"] = float(occ[pi99 == look].sum())

    # --- model-free: Q-learning -----------------------------------------
    jobs = [(g, s) for g in GAMMAS for s in range(SEEDS)]
    with Pool(os.cpu_count()) as pool:
        res["qlearning"] = pool.map(train, jobs)
    for r in res["qlearning"]:
        print(f"Q-learning gamma={r['gamma']} seed={r['seed']}: "
              f"{r['test']['people']:.3f} people ({r['seconds']:.0f}s)")

    with open(f"{OUT}/single_results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    make_figures(res)
    try:
        from single_report import build_pdf
        build_pdf(res, OUT)
    except ImportError:
        pass


def make_figures(res):
    col = {"Random": "#B0B0B0", "LOOK": "#E69F00", "ETA": "#0072B2",
           "Optimal": "#009E73", "Q-learning": "#D55E00"}

    # bar chart of the policies
    labels, vals, errs, cols = [], [], [], []
    def add(lbl, v, e, c):
        labels.append(lbl); vals.append(v); errs.append(e); cols.append(c)
    add("Random", res["random"]["people"], 0, col["Random"])
    add("LOOK", res["look"]["people"], 0, col["LOOK"])
    for g in GAMMAS:
        add(f"Optimal\n(gamma {g})", res["optimal"][str(g)]["test"]["people"], 0, col["Optimal"])
    for g in GAMMAS:
        v = [r["test"]["people"] for r in res["qlearning"] if r["gamma"] == g]
        add(f"Q-learning\n(gamma {g})", np.mean(v), np.std(v), col["Q-learning"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(labels, vals, yerr=errs, color=cols, capsize=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Average people in the system")
    ax.set_title("E0: one elevator, 5 floors (lower is better)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/e0_policies.png", dpi=150)
    plt.close(fig)

    # learning curves
    fig, ax = plt.subplots(figsize=(6, 3.3))
    for g, ls in zip(GAMMAS, ("-", "--")):
        runs = [r for r in res["qlearning"] if r["gamma"] == g]
        eps = [c["episode"] for c in runs[0]["curve"]]
        m = np.mean([[c["people"] for c in r["curve"]] for r in runs], axis=0)
        ax.plot(eps, m, ls, marker="o", ms=3, color=col["Q-learning"], label=f"Q-learning, gamma {g}")
    ax.axhline(res["optimal"]["0.99"]["test"]["people"], color=col["Optimal"], label="Optimal (value iteration)")
    ax.axhline(res["look"]["people"], color=col["LOOK"], ls=":", label="LOOK")
    ax.set_xlabel("Training episodes (50 steps each)")
    ax.set_ylabel("Average people in the system")
    ax.set_title("Learning one elevator from experience")
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/e0_learning.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
