"""
Passenger arrival patterns.

Passengers arrive as a Poisson process: in each tick the number of new
passengers is Poisson(rate). Where they start and where they go depends
on the traffic profile:

* up_peak    : morning rush. Most people start at the lobby (floor 0) and go up.
* down_peak  : evening rush. Most people go from upper floors to the lobby.
* interfloor : lunchtime. Random floor to random floor.

Up-peak is the setting of the professor's fairness example: a big cluster
at the lobby plus a few people scattered on upper floors.
"""

import numpy as np
from .simulator import Passenger

PROFILES = ("up_peak", "down_peak", "interfloor")


def arrival_rate(n_floors, n_cars, intensity):
    """
    Passengers per tick, scaled so that buildings of different sizes are
    equally busy. A round trip takes roughly (n_floors + 4) ticks, so
    this keeps the load per car comparable across experiments.
    """
    return intensity * n_cars / (n_floors + 4)


class Traffic:
    def __init__(self, profile, rate, lobby_share=0.85):
        assert profile in PROFILES, profile
        self.profile = profile
        self.rate = rate
        self.lobby_share = lobby_share

    def sample(self, t, F, rng):
        out = []
        for _ in range(rng.poisson(self.rate)):
            o, d = self._trip(F, rng)
            out.append(Passenger(o, d, t))
        return out

    def _trip(self, F, rng):
        lobby_trip = rng.random() < self.lobby_share
        if self.profile == "up_peak" and lobby_trip:
            return 0, int(rng.integers(1, F))
        if self.profile == "down_peak" and lobby_trip:
            return int(rng.integers(1, F)), 0
        # interfloor trip (also the remaining share of the peak profiles)
        o = int(rng.integers(0, F))
        d = int(rng.integers(0, F - 1))
        if d >= o:
            d += 1
        return o, d
