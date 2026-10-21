# Multi-Elevator Dispatching with Tabular Q-Learning

RL course project. A building has several elevators. Whenever someone presses a
hall button, a dispatcher must decide which elevator answers. We compare classical
dispatching rules with a dispatcher learned by tabular Q-learning, and study:

1. What does the learned policy achieve compared with heuristics?
2. What happens as the building gets taller?
3. What happens as more elevators are added?
4. How can we trade efficiency (low average wait) against fairness (nobody waits very long)?
5. How does the discount factor (planning horizon) affect learning?

Experiment E0 (one elevator) is a separate starting point: on a 5-floor building
small enough to solve exactly, it compares LOOK with the true optimal policy
(value iteration) and with Q-learning. See `results/single/single_elevator_results.pdf`.

## Quick start

```
pip install -r requirements.txt
python run_experiments.py --quick     # smoke test, about 2 minutes
python run_experiments.py             # full experiments, about 45 minutes on 2 cores
python make_report.py                 # figures + results/elevator_rl_results.pdf
python run_single.py                  # E0, one elevator, about 5 minutes
```

## Files

| File | What it does |
|---|---|
| `elevator/simulator.py` | Building simulator. Each car runs LOOK on its own stops; a dispatcher assigns new hall calls to cars. Also defines the waiting cost (reward). |
| `elevator/traffic.py` | Passenger arrivals: up-peak, down-peak, interfloor. |
| `elevator/dispatchers.py` | Baselines: Random, Nearest car, ETA (estimated time of arrival). |
| `elevator/qlearning.py` | State encoding and the tabular Q-learning dispatcher. |
| `elevator/metrics.py` | Mean / 95th percentile / max waiting time and Jain's fairness index. |
| `run_experiments.py` | Runs experiments E1 to E5 and saves `results/results.json`. |
| `make_report.py` | Draws the figures and writes the results PDF. |
| `report_text.py` | The written findings used in the PDF. |
| `elevator/single_elevator.py` | E0: the 1200-state single-elevator MDP (Yuan, Busoniu and Babuska, IFAC 2008), exact model, MRP policy evaluation, value iteration, LOOK and Q-learning. |
| `run_single.py`, `single_report.py` | Run E0 and write its figures and PDF to `results/single/`. |
| `analyze_policy.py` | Compares the learned policy's choices with ETA's on the 20-floor building. |

## The MDP in one paragraph

**Decision points** are the moments a new hall call appears. **Action**: which car
takes it. **State**: is the call at the lobby, its direction, how many hall calls are
open, whether anyone has waited more than 40 ticks, and every car's *type*
(how soon it could arrive, how many stops it already has). Cars are interchangeable,
so the state stores the car types as a sorted list, which shrinks the Q-table a lot.
**Reward**: minus the waiting cost paid until the next decision. With fairness
weight `lambda`, a passenger who has waited `w` ticks costs
`(1 - lambda) + lambda * (2w + 1) / 20` per tick, which sums to total waiting time
(`lambda = 0`) or total squared waiting time (`lambda = 1`). Squaring makes one very
long wait cost more than many short ones, which is what makes the policy fairer.

**Update rule** (tau = ticks between two decisions):

`Q(s,a) <- Q(s,a) + alpha * [ R + gamma^tau * max_a' Q(s',a') - Q(s,a) ]`

## Reproducibility

Each setting is trained with 3 seeds. All methods are tested on the same 20 passenger
streams, which are never used for training. All randomness is seeded, so rerunning
gives the same numbers.
