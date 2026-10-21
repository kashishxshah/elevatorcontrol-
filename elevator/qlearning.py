"""
Tabular Q-learning dispatcher.

The MDP
-------
* Decision points: every time a NEW hall call appears.
* Action a: which car gets the call.
* State s: a compact, discretised summary of the building (encode_state).

Using symmetry to shrink the table
----------------------------------
All cars are identical, so "car 1 is close and car 2 is far" is the same
situation as "car 2 is close and car 1 is far". We therefore describe
each car by a TYPE = (ETA bucket, busyness bucket) and the state lists
the car types in sorted order, ignoring which car is which. The action
is then "send a car of type k" (if several cars share that type, one of
them is picked at random, since they are interchangeable).
This cuts the number of states from 12^N to the number of multisets of
N types out of 12, e.g. 1,728 -> 364 for 3 cars.
* Reward: minus the waiting cost paid between this decision and the next
  one (Building.tick_cost). Maximising reward = minimising waiting.

Why Q-learning and not value iteration?
---------------------------------------
Value iteration needs the transition probabilities P(s' | s, a). Here they
depend on random passenger arrivals and on how the cars move, and writing
them down for thousands of states is impractical. Q-learning learns
Q(s, a) directly from simulated experience instead:

    Q(s,a) <- Q(s,a) + alpha * [ target - Q(s,a) ]
    target  = R + gamma^tau * max_a' Q(s', a')

The time between two decisions (tau ticks) is not constant, so the future
is discounted by gamma^tau, and R is the discounted cost summed over
those tau ticks. With tau = 1 this is the usual one-step Q-learning rule.
"""

from math import comb
import numpy as np
from .dispatchers import eta, eta_dispatcher

ETA_EDGES = (3, 8, 16)      # ETA buckets: 0-3, 4-8, 9-16, 17+ ticks
BUSY_EDGES = (0, 2)         # stops already planned: 0, 1-2, 3+
LOAD_EDGES = (1, 3, 6)      # hall buttons currently lit: 0-1, 2-3, 4-6, 7+
N_TYPES = (len(ETA_EDGES) + 1) * (len(BUSY_EDGES) + 1)   # 12 car types


def bucket(x, edges):
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def car_types(b, call):
    """Type id (0..11) of every car with respect to this hall call."""
    return [bucket(eta(c, call, b.DWELL), ETA_EDGES) * (len(BUSY_EDGES) + 1)
            + bucket(len(c.targets()), BUSY_EDGES) for c in b.cars]


def encode_state(b, call):
    """
    s = (is the call at the lobby?, call direction, how busy the building is,
         someone overdue?, sorted tuple of the car types)

    The exact call floor is NOT in the state: the ETA buckets already say
    where the cars are relative to the call, and leaving the floor out
    means every state is visited far more often, so it is learned faster.

    The building-load bucket matters a lot: future waiting cost depends
    mostly on how many people are already queueing. Without it, states
    that look identical have very different values, and the small
    difference between actions gets lost in that noise.
    """
    f, d = call
    types = car_types(b, call)
    load = bucket(len(b.call_owner), LOAD_EDGES)
    return (int(f == 0), d, load, b.overdue(), tuple(sorted(types))), types


def state_space_size(n_cars, use_symmetry=True):
    """Number of possible states with this encoding (independent of height)."""
    calls = 3                             # lobby-up, other-up, other-down
    if use_symmetry:
        car_part = comb(N_TYPES + n_cars - 1, n_cars)   # multisets of car types
    else:
        car_part = N_TYPES ** n_cars
    return calls * (len(LOAD_EDGES) + 1) * 2 * car_part


class QLearningDispatcher:
    def __init__(self, n_cars, gamma=0.99, eps_start=0.3, eps_end=0.02,
                 alpha_power=0.6, alpha_min=0.02, seed=0):
        self.N = n_cars
        self.gamma = gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.alpha_power, self.alpha_min = alpha_power, alpha_min
        self.rng = np.random.default_rng(seed)
        self.Q = {}             # state -> array of values, one per car TYPE
        self.visits = {}        # state -> array of visit counts per car type
        self.eps = eps_start
        self.training = True
        self.prev = None        # (state, action, tick) of the last decision
        self.unseen_decisions = 0
        self.total_decisions = 0

    # ------------------------------------------------------------------
    def __call__(self, b, call):
        s, types = encode_state(b, call)
        available = sorted(set(types))          # car types present right now
        if self.training:
            self._learn_from_last_decision(b, s)
            a = self._epsilon_greedy(s, available)
            self.prev = (s, a, b.t)
        else:
            a = self._greedy_or_fallback(b, call, s, available)
            if a is None:
                return eta_dispatcher(b, call)
        # send one of the cars of the chosen type
        cars = [i for i, t in enumerate(types) if t == a]
        return int(self.rng.choice(cars))

    def _epsilon_greedy(self, s, available):
        """With probability eps explore a random car type, otherwise exploit."""
        q = self.Q.get(s)
        if q is None or self.rng.random() < self.eps:
            return int(self.rng.choice(available))
        vals = q[available]
        best = [a for a, v in zip(available, vals) if v == vals.max()]
        return int(self.rng.choice(best))

    def _greedy_or_fallback(self, b, call, s, available):
        """
        At test time pick the best action among those tried in training.
        A state never seen in training has no learned values, so we fall
        back to the ETA rule (return None) and count how often that
        happens. That count measures how much of the state space the
        table actually covers.
        """
        self.total_decisions += 1
        q, n = self.Q.get(s), self.visits.get(s)
        tried = [] if n is None else [a for a in available if n[a] > 0]
        if not tried:
            self.unseen_decisions += 1
            return None
        vals = q[tried]
        best = [a for a, v in zip(tried, vals) if v == vals.max()]
        return int(self.rng.choice(best))

    def _state_value(self, s):
        """max_a Q(s, a) over the actions AVAILABLE in s, i.e. the car types
        present there (the last part of the state). Q starts at 0 for every
        action, so an unseen state contributes 0."""
        q = self.Q.get(s)
        return 0.0 if q is None else float(q[sorted(set(s[-1]))].max())

    def _learn_from_last_decision(self, b, s_next):
        """Update the pending transition when the next decision arrives in s_next."""
        if self.prev is not None:
            self._update(b, self._state_value(s_next))

    def end_episode(self, b):
        """
        Update the pending transition when the episode ends.

        Without this, the last decision of every episode would never be
        learned from. The episode end is an artificial time limit, not a
        real terminal state (the building keeps running), so we do NOT
        treat the future as worth 0. We add the cost paid up to the end and
        bootstrap from the value of the last decision state, our best
        available estimate of the building's value at the cut-off, because
        no new decision state exists there.
        """
        if self.training and self.prev is not None:
            s = self.prev[0]
            self._update(b, self._state_value(s))
        self.prev = None

    def _update(self, b, future):
        s, a, t0 = self.prev
        tau = b.t - t0
        if tau <= 0:
            return
        costs = np.asarray(b.cost_log[t0:b.t])
        R = -float(np.sum(costs * self.gamma ** np.arange(tau)))
        target = R + self.gamma ** tau * future

        q = self.Q.setdefault(s, np.zeros(N_TYPES))
        n = self.visits.setdefault(s, np.zeros(N_TYPES))
        n[a] += 1
        # learning rate shrinks as a (state, action) pair is visited more
        alpha = max(self.alpha_min, n[a] ** -self.alpha_power)
        q[a] += alpha * (target - q[a])

    # ------------------------------------------------------------------
    def start_episode(self):
        self.prev = None

    def set_progress(self, frac):
        """Decay exploration linearly over the first 80% of training."""
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * min(1.0, frac / 0.8)

    def eval_mode(self):
        self.training = False
        self.unseen_decisions = 0
        self.total_decisions = 0

    @property
    def coverage(self):
        """Share of test decisions made in states seen during training."""
        if self.total_decisions == 0:
            return 1.0
        return 1 - self.unseen_decisions / self.total_decisions
