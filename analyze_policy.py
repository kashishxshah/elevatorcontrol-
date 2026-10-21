"""
Two diagnostics behind the explanations in the report.

1. What does the learned policy do differently from ETA? (20 floors, up-peak)
   Trains one Q-learning dispatcher, then on test streams compares each of its
   choices with the car ETA would have picked.

2. How different are the learned values of different actions? (10 floors)
   For up-peak and interfloor traffic, measures in well-visited states how far
   apart the Q-values of the available car types are, relative to their size.
   Small gaps mean the agent cannot tell good and bad choices apart.

    python analyze_policy.py      # writes results/policy_analysis.json
"""

import json

import numpy as np

from run_experiments import make_building, EP_TICKS, TEST_SEED0
from elevator.qlearning import QLearningDispatcher
from elevator.dispatchers import eta_dispatcher, eta


def train(cfg, episodes, seed=0):
    agent = QLearningDispatcher(cfg["cars"], gamma=cfg["gamma"], seed=seed)
    for ep in range(episodes):
        agent.set_progress(ep / episodes)
        agent.start_episode()
        b = make_building(cfg, seed * 1_000_000 + ep)
        b.run(agent, EP_TICKS)
        agent.end_episode(b)    # learn from the last pending decision too
    return agent


def compare_with_eta(episodes=3000, n_test=5):
    cfg = dict(floors=20, cars=3, profile="up_peak", lam=0.0, gamma=0.99)
    agent = train(cfg, episodes)
    agent.eval_mode()
    log = dict(agree=[], q_stops=[], e_stops=[], q_eta=[], e_eta=[])

    def compare(b, call):
        a, e = agent(b, call), eta_dispatcher(b, call)
        log["agree"].append(a == e)
        log["q_stops"].append(len(b.cars[a].targets()))
        log["e_stops"].append(len(b.cars[e].targets()))
        log["q_eta"].append(eta(b.cars[a], call, b.DWELL))
        log["e_eta"].append(eta(b.cars[e], call, b.DWELL))
        return a                         # the learned policy is the one acting

    for i in range(n_test):
        make_building(cfg, TEST_SEED0 + i).run(compare, EP_TICKS)
    return {k: float(np.mean(v)) for k, v in log.items()}


def value_gaps(profile, episodes=3000, min_visits=30):
    """Median relative gap between the best and worst available action's Q-value."""
    cfg = dict(floors=10, cars=3, profile=profile, lam=0.0, gamma=0.99)
    agent = train(cfg, episodes)
    gaps = []
    for s, q in agent.Q.items():
        n = agent.visits[s]
        acts = [a for a in sorted(set(s[-1])) if n[a] >= min_visits]
        if len(acts) >= 2:
            v = q[acts]
            gaps.append((v.max() - v.min()) / abs(v.mean()))
    return dict(median_gap=float(np.median(gaps)), states=len(gaps))


def main():
    out = dict(eta_comparison_20_floors=compare_with_eta(),
               value_gap_up_peak=value_gaps("up_peak"),
               value_gap_interfloor=value_gaps("interfloor"))
    c = out["eta_comparison_20_floors"]
    print(f"Same car as ETA: {100 * c['agree']:.0f}% of calls")
    print(f"Stops already planned for the chosen car: learned {c['q_stops']:.2f}, ETA {c['e_stops']:.2f}")
    print(f"Estimated arrival of the chosen car: learned {c['q_eta']:.1f}, ETA {c['e_eta']:.1f} ticks")
    for p in ("up_peak", "interfloor"):
        g = out[f"value_gap_{p}"]
        print(f"{p}: median gap between best and worst action value "
              f"{100 * g['median_gap']:.1f}% ({g['states']} well-visited states)")
    with open("results/policy_analysis.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
