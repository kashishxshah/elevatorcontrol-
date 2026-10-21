"""
Written findings for the results PDF.

Numbers are pulled from results/results.json and results/policy_analysis.json,
so the text stays consistent with the tables. The qualitative statements were
written for the final run (fixed Q-learning update, clean results file); if a
rerun changes the picture, update the wording here.
"""

import json
import os


def make_text(R):
    E = {k: sorted(v) for k, v in R.exps.items()}
    up, down, inter = [next(k for k in E["E1"] if k[2] == p)
                       for p in ("up_peak", "down_peak", "interfloor")]
    g = lambda k, m="Q-learning", x="mean": R.get(k, m, x)[0]
    sd = lambda k, x="mean": R.get(k, "Q-learning", x)[1]
    imp = R.improvement

    floors = sorted(E["E2"], key=lambda k: k[0])
    cars = sorted(E["E3"], key=lambda k: k[1])
    lam = sorted(E["E4"], key=lambda k: k[3])
    gam = sorted(E["E5"], key=lambda k: k[4])
    f_lo, f_hi = floors[0], floors[-1]

    pa_path = "results/policy_analysis.json"
    pa = json.load(open(pa_path)) if os.path.exists(pa_path) else None

    T = {}
    e1 = (
        f"Under up-peak traffic Q-learning beats ETA: mean wait "
        f"{g(up):.2f} +/- {sd(up):.2f} vs {g(up, 'ETA'):.2f} ticks ({imp(up):+.0f}%), and 95th "
        f"percentile {g(up, x='p95'):.1f} vs {g(up, 'ETA', 'p95'):.1f}. The gain holds in all "
        "three seeds, but its size at 10 floors was sensitive to small corrections to the "
        "training code during development (it ranged from about +4% to +12%), so we read it as "
        "a modest advantage rather than a precise figure. Under down-peak and interfloor "
        f"traffic it is clearly worse than ETA ({g(down):.1f} vs {g(down, 'ETA'):.1f} and "
        f"{g(inter):.1f} vs {g(inter, 'ETA'):.1f} ticks), close to the Random dispatcher "
        f"({g(down, 'Random'):.1f} and {g(inter, 'Random'):.1f}). "
        "Our explanation is credit assignment. The reward is the waiting cost of everyone in "
        "the building. In up-peak most calls come from the lobby, the same states recur, and "
        "each decision decides where a whole group goes. When calls are spread over all floors, "
        "each decision affects one or two people, and that small effect is buried in the random "
        "fluctuation of the building-wide cost.")
    if pa:
        e1 += (f" The diagnostic in analyze_policy.py supports this: in well-visited states the "
               f"gap between the best and worst available action's value has a median of "
               f"{100 * pa['value_gap_interfloor']['median_gap']:.1f}% of the value under "
               f"interfloor traffic, against {100 * pa['value_gap_up_peak']['median_gap']:.1f}% "
               "under up-peak, so in interfloor traffic the greedy choice is close to a coin toss.")
    T["E1"] = e1 + (" Nearest car does badly in up-peak because it ignores the stops a car has "
                    "already committed to.")

    e2 = (
        f"The advantage of learning grows with building height: from {imp(f_lo):+.0f}% at "
        f"{f_lo[0]} floors to {imp(f_hi):+.0f}% at {f_hi[0]} floors. ETA's mean wait rises "
        f"from {g(f_lo, 'ETA'):.1f} to {g(f_hi, 'ETA'):.1f} ticks while Q-learning's rises only "
        f"from {g(f_lo):.1f} to {g(f_hi):.1f}; the 95th percentile at {f_hi[0]} floors is "
        f"{g(f_hi, x='p95'):.1f} vs {g(f_hi, 'ETA', 'p95'):.1f} ticks. ETA is greedy: it "
        "minimises the arrival time for the caller in front of it, while Q-learning minimises "
        "the discounted waiting of everyone, and the long round trips in a tall building make "
        "those two goals differ more.")
    if pa:
        c = pa["eta_comparison_20_floors"]
        e2 += (f" Comparing choices on the 20-floor building (analyze_policy.py), the learned "
               f"policy picks the same car as ETA on {100 * c['agree']:.0f}% of calls. Averaged over "
               f"all calls, the car it sends is expected to arrive later ({c['q_eta']:.1f} vs "
               f"{c['e_eta']:.1f} ticks) and has a similar number of stops already planned "
               f"({c['q_stops']:.2f} vs {c['e_stops']:.2f}). So it accepts a slower pickup for "
               "some callers in exchange for shorter waits overall; we could not reduce what it "
               "learned to one simple rule.")
    T["E2"] = e2 + (" With only 6 floors trips are short and ETA is as good or better. Because "
                    "our state describes cars relative to the call, the Q-table does not grow "
                    f"with height ({g(f_lo, x='states_visited'):,.0f} to "
                    f"{g(f_hi, x='states_visited'):,.0f} states visited).")

    T["E3"] = (
        f"With {cars[0][1]} cars there is little room for clever assignment and Q-learning "
        f"roughly matches ETA ({imp(cars[0]):+.0f}%). With more cars Q-learning is ahead ("
        + ", ".join(f"{imp(k):+.0f}% with {k[1]}" for k in cars[1:]) +
        f"), but the gain does not keep growing. The cost shows up in learning speed: the number of states visited grows from "
        f"{g(cars[0], x='states_visited'):,.0f} to {g(cars[-1], x='states_visited'):,.0f}, and "
        f"the {cars[-1][1]}-car agent learns slowest (its learning curve starts far above ETA "
        "and only drops below it near the end of training). This is the curse of "
        "dimensionality: each extra car multiplies the situations the table must learn. "
        "Treating cars as interchangeable types keeps it manageable: labelling each car "
        f"separately would allow about {12 ** cars[-1][1] * 24:,} states for {cars[-1][1]} cars "
        f"instead of {R.get(cars[-1], 'Q-learning', 'state_space')[0]:,.0f}.")

    T["E4"] = (
        "Raising lambda gives a small improvement in fairness at almost no cost in efficiency. "
        "Jain's index rises steadily ("
        + ", ".join(f"{g(k, x='jain'):.3f}" for k in lam) +
        " for lambda = " + ", ".join(f"{k[3]:g}" for k in lam) +
        f"), and the maximum wait falls from {g(lam[0], x='max'):.1f} +/- {sd(lam[0], 'max'):.1f} "
        f"to {g(lam[-1], x='max'):.1f} +/- {sd(lam[-1], 'max'):.1f} ticks, while the mean wait "
        f"stays between {min(g(k) for k in lam):.1f} and {max(g(k) for k in lam):.1f}. The effect "
        "is modest and partly within seed-to-seed variation (the maximum wait is not monotone in "
        "lambda), and an earlier run during development showed no clear effect at all, so we "
        "treat it as weak evidence. The likely reason it is small is structural: the dispatcher "
        "only acts when a new call appears and can never reassign an old one, so it has little "
        "power to rescue someone who is already waiting. All Q-learning settings have a much "
        f"shorter worst wait than ETA, whose maximum wait here is {g(lam[0], 'ETA', 'max'):.1f} "
        "ticks. Note that Jain's index measures how evenly waits are spread, not how short they "
        "are.")

    T["E5"] = (
        "A very short horizon hurts. Mean wait is "
        + ", ".join(f"{g(k):.2f} +/- {sd(k):.2f}" for k in gam) +
        " ticks for gamma = " + ", ".join(f"{k[4]:g}" for k in gam) +
        f". With gamma = {gam[0][4]:g} (about {1 / (1 - gam[0][4]):.0f} ticks of look-ahead) the "
        "agent barely beats ETA and varies a lot between seeds; gamma = "
        f"{gam[1][4]:g} and {gam[2][4]:g} are both clearly better and within noise of each other. "
        "Our interpretation: a short horizon misses delayed effects such as overloading a busy "
        "car. Note that the gamma = 0.95 vs 0.99 ranking changed between development runs, so "
        "we do not claim either is best.")

    T["conclusions"] = [
        "RL is worth it where the classical rule is short-sighted and decisions have large, "
        f"delayed consequences: up-peak in tall buildings ({imp(f_hi):.0f}% lower mean wait than "
        f"ETA at {f_hi[0]} floors). With a short building, few cars, or one elevator (E0), ETA "
        "or LOOK is already about as good.",
        "Tabular Q-learning with a building-wide reward fails when each decision has only a small "
        "effect (interfloor and down-peak traffic): a credit-assignment problem, not a shortage "
        "of data.",
        "Taller buildings clearly increase the benefit of learning. More cars help a little, but "
        "slow learning down as the state space grows; exploiting symmetry between cars keeps "
        "the table manageable.",
        "Weighting long waits more heavily in the reward gave only a small fairness gain, because "
        "assignment-only control cannot revisit old calls. Fairness needs actions that can "
        "prioritise long waiters, not just a different reward.",
        f"A very short planning horizon (gamma = {gam[0][4]:g}) clearly hurts; beyond about 20 "
        "ticks of look-ahead the differences were within noise.",
    ]
    T["limitations"] = [
        "Simplified simulator: fixed travel and door times, Poisson traffic, only the assigned "
        "car picks up a hall call, and calls are never reassigned.",
        "Coarse, hand-designed state (ETA and busyness buckets). States unseen in training fall "
        "back to ETA, although coverage was above 99% in all reported settings.",
        "One traffic intensity, three seeds per setting, and buildings of at most 20 floors and "
        "5 cars because of compute. The smaller effects (the 10-floor up-peak gain, E4, E5) are "
        "close to the seed-to-seed variation and moved between development runs; the large "
        "effects (tall buildings, down-peak and interfloor losses) were stable.",
        "The agent is trained separately for each traffic profile and building, and is not "
        "tested on traffic that changes during the day.",
    ]
    T["further"] = [
        "Function approximation (e.g. a small neural network or linear features) to replace the "
        "table and scale to more cars.",
        "Better credit assignment: per-car or per-passenger rewards instead of a single "
        "building-wide reward.",
        "Richer actions for fairness: reassigning waiting calls or explicit priority for long "
        "waiters, and constrained formulations (minimise mean wait subject to a maximum wait).",
        "Real traffic traces, time-varying traffic, and destination dispatch, where passengers "
        "enter their destination in the lobby.",
    ]
    return T
