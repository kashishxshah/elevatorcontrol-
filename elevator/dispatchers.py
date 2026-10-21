"""
Baseline dispatchers (no learning).

Every dispatcher is a function  dispatcher(building, call) -> car index,
called whenever a new hall call (floor, direction) appears.
Each car then serves its calls using LOOK (see simulator.py), so these
baselines are "LOOK + a classical assignment rule".
"""

import numpy as np
from .simulator import UP, DIR_SIGN


def eta(car, call, dwell):
    """
    Estimated ticks until `car` reaches the hall call, assuming it follows
    LOOK through its current stops. Each stop it already has adds `dwell`.
    """
    f, d = call
    p, s = car.floor, car.direction
    targets = car.targets()
    n_stops = len(targets)

    if s == 0:
        dist = abs(p - f)
    else:
        if s == -1:                     # mirror the building so we only
            p, f = -p, -f               # have to reason about going up
            targets = {-x for x in targets}
            d = 1 - d
        top = max(targets | {p})
        bottom = min(targets | {p})
        if d == UP and f >= p:          # call is ahead, same direction
            dist = f - p
        elif d != UP:                   # call needs the opposite direction
            turn = max(top, f)
            dist = (turn - p) + (turn - f)
        else:                           # same direction but behind us
            turn_hi = max(top, p)
            turn_lo = min(bottom, f)
            dist = (turn_hi - p) + (turn_hi - turn_lo) + (f - turn_lo)
    return dist + dwell * n_stops


def random_dispatcher(rng):
    def choose(b, call):
        return int(rng.integers(b.N))
    return choose


def nearest_car(rng):
    """Send the car that is physically closest, ignoring where it is going.
    Ties (e.g. two cars parked at the lobby) are broken at random."""
    def choose(b, call):
        d = np.array([abs(c.floor - call[0]) for c in b.cars])
        return int(rng.choice(np.flatnonzero(d == d.min())))
    return choose


def eta_dispatcher(b, call):
    """Send the car with the smallest estimated time of arrival (industry classic)."""
    return int(np.argmin([eta(c, call, b.DWELL) for c in b.cars]))


BASELINES = {
    "Nearest car": nearest_car,
    "ETA": lambda rng: eta_dispatcher,
    "Random": random_dispatcher,
}
