"""
Runs every experiment and saves the raw numbers to results/results.json.

    python run_experiments.py            # full run (about an hour on 2 cores)
    python run_experiments.py --quick    # small smoke test (a few minutes)

Experiments
-----------
E1  Traffic patterns : RL vs baselines on 10 floors, 3 cars, three profiles
E2  Building height  : up-peak, 3 cars, 6 / 10 / 15 / 20 floors
E3  Number of cars   : up-peak, 10 floors, 2 / 3 / 4 / 5 cars
E4  Fairness         : up-peak, 20 floors, 3 cars, fairness lambda 0 .. 1
                       (the 20-floor building is used because long waits
                       actually occur there, so fairness has something to fix)
E5  Discount factor  : up-peak, 10 floors, 3 cars, gamma 0.9 / 0.95 / 0.99

Fair comparison: every method is tested on exactly the same 20 random
passenger streams (test seeds), which are never used in training.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from elevator.simulator import Building
from elevator.traffic import Traffic, arrival_rate
from elevator.dispatchers import BASELINES
from elevator.qlearning import QLearningDispatcher, state_space_size
from elevator.metrics import episode_waits, summarize

INTENSITY = 2.4         # traffic level (see traffic.arrival_rate)
EP_TICKS = 800          # length of one episode
WARMUP = 100            # ignore passengers who arrive before this tick
TEST_SEED0 = 100_000    # test streams use seeds 100000, 100001, ...


def make_building(cfg, seed):
    rate = arrival_rate(cfg["floors"], cfg["cars"], INTENSITY)
    return Building(cfg["floors"], cfg["cars"], Traffic(cfg["profile"], rate),
                    fairness_lambda=cfg.get("lam", 0.0), seed=seed)


def evaluate(cfg, dispatcher, n_test):
    """Average metrics over the fixed test streams."""
    runs = []
    for i in range(n_test):
        b = make_building(cfg, TEST_SEED0 + i)
        b.run(dispatcher, EP_TICKS)
        runs.append(summarize(episode_waits(b, WARMUP)))
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def run_baseline(job):
    cfg, name, n_test = job
    disp = BASELINES[name](np.random.default_rng(0))
    return dict(cfg=cfg, method=name, seed=None, metrics=evaluate(cfg, disp, n_test))


def run_qlearning(job):
    cfg, seed, n_episodes, n_test = job
    t0 = time.time()
    agent = QLearningDispatcher(cfg["cars"], gamma=cfg.get("gamma", 0.99), seed=seed)
    curve = []
    checkpoints = set(np.linspace(0, n_episodes, 7).astype(int)[1:])
    for ep in range(n_episodes):
        agent.set_progress(ep / n_episodes)
        agent.start_episode()
        # training streams use seeds that never overlap the test seeds
        b = make_building(cfg, seed * 1_000_000 + ep)
        b.run(agent, EP_TICKS)
        agent.end_episode(b)    # learn from the last pending decision too
        if ep + 1 in checkpoints:
            agent.eval_mode()
            m = evaluate(cfg, agent, 5)
            curve.append(dict(episode=ep + 1, mean=m["mean"], coverage=agent.coverage))
            agent.training = True

    agent.eval_mode()
    metrics = evaluate(cfg, agent, n_test)
    metrics["coverage"] = agent.coverage
    metrics["states_visited"] = len(agent.Q)
    metrics["state_space"] = state_space_size(cfg["cars"])
    metrics["train_seconds"] = time.time() - t0
    return dict(cfg=cfg, method="Q-learning", seed=seed, metrics=metrics, curve=curve)


def experiment_configs():
    base = dict(floors=10, cars=3, profile="up_peak", lam=0.0, gamma=0.99)
    exps = {"E1": [dict(base, profile=p) for p in ("up_peak", "down_peak", "interfloor")],
            "E2": [dict(base, floors=f) for f in (6, 10, 15, 20)],
            "E3": [dict(base, cars=n) for n in (2, 3, 4, 5)],
            "E4": [dict(base, floors=20, lam=l) for l in (0.0, 0.25, 0.5, 0.75, 1.0)],
            "E5": [dict(base, gamma=g) for g in (0.9, 0.95, 0.99)]}
    for name, cfgs in exps.items():
        for c in cfgs:
            c["experiment"] = name
    return exps


def key(cfg):
    """Identical settings in different experiments are only run once."""
    return (cfg["floors"], cfg["cars"], cfg["profile"], cfg["lam"], cfg["gamma"])


def rerun_baselines(args):
    with open(args.out) as fh:
        data = json.load(fh)
    q_records = [r for r in data["records"] if r["method"] == "Q-learning"]
    cfgs = {(r["cfg"]["floors"], r["cfg"]["cars"], r["cfg"]["profile"]): r["cfg"] for r in q_records}
    jobs = [(c, name, data["settings"]["tests"]) for c in cfgs.values() for name in BASELINES]
    with Pool(args.workers) as pool:
        data["records"] = pool.map(run_baseline, jobs) + q_records
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"recomputed {len(jobs)} baseline runs in {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tests", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--add", action="store_true",
                    help="only run settings missing from an existing results file")
    ap.add_argument("--baselines-only", action="store_true",
                    help="recompute only the baselines inside an existing results file")
    args = ap.parse_args()
    if args.baselines_only:
        return rerun_baselines(args)
    if args.quick:
        args.episodes, args.seeds, args.tests = 300, 1, 5

    exps = experiment_configs()
    unique = {}
    for cfgs in exps.values():
        for c in cfgs:
            unique.setdefault(key(c), c)

    old_records = []
    if args.add:
        with open(args.out) as fh:
            old = json.load(fh)
        old_records = old["records"]
        done = {key(r["cfg"]) for r in old_records if r["method"] == "Q-learning"}
        done_b = {(r["cfg"]["floors"], r["cfg"]["cars"], r["cfg"]["profile"])
                  for r in old_records if r["method"] != "Q-learning"}
        unique = {k: c for k, c in unique.items() if k not in done}

    q_jobs = [(c, s, args.episodes, args.tests) for c in unique.values()
              for s in range(args.seeds)]
    # baselines do not learn, so lambda and gamma do not change them
    base_cfgs = {(c["floors"], c["cars"], c["profile"]): c for c in unique.values()}
    if args.add:
        base_cfgs = {k: c for k, c in base_cfgs.items() if k not in done_b}
    b_jobs = [(c, name, args.tests) for c in base_cfgs.values() for name in BASELINES]

    print(f"{len(q_jobs)} Q-learning runs x {args.episodes} episodes, "
          f"{len(b_jobs)} baseline runs, {args.workers} workers")
    t0 = time.time()
    with Pool(args.workers) as pool:
        records = pool.map(run_baseline, b_jobs)
        # biggest jobs first so the pool finishes evenly
        q_jobs.sort(key=lambda j: -(j[0]["cars"] * j[0]["floors"]))
        for i, r in enumerate(pool.imap_unordered(run_qlearning, q_jobs), 1):
            c = r["cfg"]
            print(f"[{i}/{len(q_jobs)}] floors={c['floors']} cars={c['cars']} "
                  f"{c['profile']} lam={c['lam']} gamma={c['gamma']} seed={r['seed']}: "
                  f"mean wait {r['metrics']['mean']:.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            records.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(settings=dict(vars(args), intensity=INTENSITY, ep_ticks=EP_TICKS,
                                     warmup=WARMUP),
                       experiments={k: [key(c) for c in v] for k, v in exps.items()},
                       records=old_records + records), fh, indent=1)
    print(f"saved {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
