"""
Performance and fairness metrics, computed from one simulated episode.

Waiting time = time from pressing the hall button to boarding a car.
People still waiting when the episode ends are counted with the wait
they have had so far, so a policy cannot hide a starved passenger.
"""

import numpy as np


def episode_waits(building, warmup):
    t_end = building.t
    waits = []
    for p in building.passengers:
        if p.t_arrive < warmup:
            continue            # skip the empty-building start
        board = p.t_board if p.t_board is not None else t_end
        waits.append(board - p.t_arrive)
    return np.array(waits, dtype=float)


def summarize(waits):
    if len(waits) == 0:
        return dict(mean=0.0, p95=0.0, max=0.0, jain=1.0, n=0)
    w = waits + 1.0             # +1 so zero waits do not break Jain's index
    return dict(
        mean=float(waits.mean()),
        p95=float(np.percentile(waits, 95)),
        max=float(waits.max()),
        # Jain's index: 1 = everyone waits the same, 1/n = one person gets all the wait
        jain=float(w.sum() ** 2 / (len(w) * (w ** 2).sum())),
        n=int(len(waits)),
    )
