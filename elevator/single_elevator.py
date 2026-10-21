"""
Single-elevator benchmark (experiment E0), the starting point of the project.

Model from Yuan, Busoniu and Babuska, "Reinforcement learning for elevator
control", IFAC World Congress 2008:

* 5 floors (0 = ground), one car, capacity 4, one time step = 2 seconds
  (one floor of travel, or one stop).
* Down-peak traffic: passengers appear on floors 1-4 and all go to the
  ground floor. Each step one event happens with probabilities
  [0.6875, 0.0625, 0.09375, 0.09375, 0.0625]: nobody, or a passenger
  on floor 1, 2, 3 or 4. At most one person waits per floor.
* State  x = (c1, c2, c3, c4, p, v, o): call flags for floors 1-4,
  position p, velocity v in {-1, 0, +1}, occupancy o. 2^4 * 5 * 3 * 5 = 1200 states.
* Action u in {-1 (down), 0 (stop), +1 (up)}. The car cannot reverse
  without stopping first, and cannot leave the building.
* Reward = -(number of people waiting + number riding).

Because there are only 1200 states, the transition probabilities can be
written down exactly. That lets us use the methods from the course
directly: policy evaluation as a Markov reward process and value
iteration for the optimal policy, and compare them with Q-learning and
the classical LOOK rule.
"""

import itertools
import numpy as np

ARRIVAL_PROBS = np.array([0.6875, 0.0625, 0.09375, 0.09375, 0.0625])
CAPACITY = 4
TOP = 4
STEP_SECONDS = 2.0
ACTIONS = (-1, 0, 1)

STATES = list(itertools.product((0, 1), (0, 1), (0, 1), (0, 1),
                                range(5), (-1, 0, 1), range(5)))
INDEX = {s: i for i, s in enumerate(STATES)}
N_STATES, N_ACTIONS = len(STATES), len(ACTIONS)


def valid_actions(state):
    *_, p, v, o = state
    ok = []
    for a, u in enumerate(ACTIONS):
        if (p == TOP and u == 1) or (p == 0 and u == -1):
            continue            # cannot leave the building
        if v * u == -1:
            continue            # must stop before changing direction
        ok.append(a)
    return ok


VALID = [valid_actions(s) for s in STATES]
MASK = np.zeros((N_STATES, N_ACTIONS), dtype=bool)
for i, acts in enumerate(VALID):
    MASK[i, acts] = True


def transition(state, u, event):
    """Deterministic part of a step: next state given the action and the arrival event."""
    calls = list(state[:4])
    p, v, o = state[4:]
    if event > 0:
        calls[event - 1] = 1                # at most one waiting person per floor
    p = min(TOP, max(0, p + u))
    if u == 0:                              # doors open
        if p == 0:
            o = 0                           # everyone leaves at the ground floor
        elif calls[p - 1] and o < CAPACITY:
            calls[p - 1] = 0                # the waiting person gets on
            o += 1
    return (*calls, p, u, o)


def reward(state):
    return -float(sum(state[:4]) + state[6])


def build_model():
    """Exact P[s, a, s'] and R[s, a]."""
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS))
    for i, s in enumerate(STATES):
        for a in VALID[i]:
            R[i, a] = reward(s)
            for e, prob in enumerate(ARRIVAL_PROBS):
                P[i, a, INDEX[transition(s, ACTIONS[a], e)]] += prob
    return P, R


# ----------------------------------------------------------------------
# solving the MDP with the model (course material: MRP and value iteration)
# ----------------------------------------------------------------------
def evaluate_policy(P, R, policy, gamma):
    """
    A fixed policy turns the MDP into a Markov reward process:
        P_pi(s'|s) = P(s'|s, pi(s)),   R_pi(s) = R(s, pi(s))
    and its values solve the Bellman expectation equation
        V = R_pi + gamma * P_pi V   ->   V = (I - gamma P_pi)^-1 R_pi
    """
    idx = np.arange(N_STATES)
    P_pi, R_pi = P[idx, policy], R[idx, policy]
    return np.linalg.solve(np.eye(N_STATES) - gamma * P_pi, R_pi)


def value_iteration(P, R, gamma, theta=1e-6, max_iter=10_000):
    """Repeat Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) max_a' Q(s',a') until it stops changing."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    for it in range(1, max_iter + 1):
        V = np.where(MASK, Q, -np.inf).max(axis=1)
        Q_new = np.where(MASK, R + gamma * P @ V, 0.0)
        delta = np.abs(Q_new - Q).max()
        Q = Q_new
        if delta < theta:
            break
    return np.where(MASK, Q, -np.inf).argmax(axis=1), Q, it


def long_run_cost(P, policy, start=0, horizon=20_000):
    """Average number of people in the system in the long run under a policy."""
    idx = np.arange(N_STATES)
    P_pi = P[idx, policy]
    dist = np.zeros(N_STATES)
    dist[start] = 1.0
    avg = np.zeros(N_STATES)
    for _ in range(horizon):
        dist = dist @ P_pi
        avg += dist
    avg /= horizon
    people = np.array([sum(s[:4]) + s[6] for s in STATES])
    return float(avg @ people)


# ----------------------------------------------------------------------
# policies that do not need learning
# ----------------------------------------------------------------------
def look_action(state):
    """
    LOOK for down-peak traffic ("down-collective"), as a function of the
    state only: when empty, go up to the HIGHEST call without stopping on
    the way, then sweep down picking people up, unload at the ground
    floor, and wait where you are when there is nothing to do.
    (Picking people up on the way up would make the car turn around
    and leave the higher callers waiting, which is unfair and slower.)
    """
    calls = state[:4]
    p, v, o = state[4:]
    here = p > 0 and calls[p - 1] == 1 and o < CAPACITY
    above = o < CAPACITY and any(calls[f - 1] for f in range(p + 1, TOP + 1))
    below = o > 0 or any(calls[f - 1] for f in range(1, p))

    if v == 1:                               # going up
        if above:
            return 1
        return 0                             # reached the highest call (or nothing left)
    if v == -1:                              # going down
        if here or (p == 0 and o > 0):
            return 0                         # stop to pick up or unload
        return -1 if below and p > 0 else 0
    # stopped
    if here and not above:
        return 0                             # load the person here
    if o > 0:
        return -1 if p > 0 else 0            # people inside: head down
    if above:
        return 1                             # empty: go up to the highest call
    if below:
        return -1
    return 0                                 # nothing to do: wait here


def table_policy(fn):
    return np.array([ACTIONS.index(fn(s)) for s in STATES])


# ----------------------------------------------------------------------
# simulation, with passengers tracked one by one for the metrics
# ----------------------------------------------------------------------
class Simulator:
    def __init__(self, seed, start=None):
        self.rng = np.random.default_rng(seed)
        # default start: empty building, car stopped at the ground floor
        self.state = STATES[start] if start is not None else (0, 0, 0, 0, 0, 0, 0)
        self.t = 0
        # people already in the start state are counted from time 0
        self.waiting = {f: 0 for f in range(1, 5) if self.state[f - 1]}   # floor -> arrival step
        self.riders = [0] * self.state[6]                                 # arrival steps of riders
        self.journeys = []       # completed journey times in seconds

    def step(self, a):
        s = self.state
        u = ACTIONS[a]
        event = self.rng.choice(5, p=ARRIVAL_PROBS)
        if event > 0 and not s[event - 1]:
            self.waiting[event] = self.t
        nxt = transition(s, u, event)
        p = nxt[4]
        if u == 0:
            if p == 0:
                self.journeys += [(self.t + 1 - t0) * STEP_SECONDS for t0 in self.riders]
                self.riders = []
            elif p in self.waiting and not nxt[p - 1]:     # the person here boarded
                self.riders.append(self.waiting.pop(p))
        r = reward(s)
        self.state = nxt
        self.t += 1
        return INDEX[nxt], r

    def mean_elapsed_now(self):
        """The paper's 'average waiting time': mean time so far of everyone in the system."""
        times = [(self.t - t0) * STEP_SECONDS for t0 in list(self.waiting.values()) + self.riders]
        return float(np.mean(times)) if times else 0.0


def run_policy(policy_fn, seed, steps=250):
    """policy_fn(state_index, simulator) -> action index. Returns the episode metrics."""
    sim = Simulator(seed)
    s, people, elapsed = INDEX[sim.state], [], []
    for _ in range(steps):
        a = policy_fn(s, sim)
        s, r = sim.step(a)
        people.append(-r)
        elapsed.append(sim.mean_elapsed_now())
    return dict(trial_wait=float(np.mean(elapsed)),
                journey=float(np.mean(sim.journeys)) if sim.journeys else float("nan"),
                people=float(np.mean(people)))


# ----------------------------------------------------------------------
# model-free learning
# ----------------------------------------------------------------------
def q_learning(episodes, steps, gamma, seed, alpha=None, eps_start=0.8, eps_end=0.02,
               exploring_starts=True, on_checkpoint=None, checkpoints=()):
    """
    Plain one-step Q-learning on the simulator, with no access to P:
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    Exploring starts: each training episode begins in a random state.
    Rare states (e.g. a full car with every floor calling) are almost
    never reached from an empty building, so without this their values
    are never learned and the greedy policy can get stuck in them.

    Learning rate: a constant alpha if given, otherwise
    alpha = max(0.02, n^-0.6) for the n-th visit of (s, a), as in the
    multi-elevator agent.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    visits = np.zeros((N_STATES, N_ACTIONS))
    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * min(1.0, ep / (0.8 * episodes))
        start = int(rng.integers(N_STATES)) if exploring_starts else None
        sim = Simulator(seed * 1_000_003 + ep, start)
        s = INDEX[sim.state]
        for _ in range(steps):
            acts = VALID[s]
            if rng.random() < eps:
                a = int(rng.choice(acts))
            else:
                a = acts[int(np.argmax(Q[s, acts]))]
            s2, r = sim.step(a)
            target = r + gamma * Q[s2, VALID[s2]].max()
            visits[s, a] += 1
            lr = alpha if alpha is not None else max(0.02, visits[s, a] ** -0.6)
            Q[s, a] += lr * (target - Q[s, a])
            s = s2
        if on_checkpoint and ep + 1 in checkpoints:
            on_checkpoint(ep + 1, greedy(Q))
    return greedy(Q), Q


def greedy(Q):
    return np.where(MASK, Q, -np.inf).argmax(axis=1)
