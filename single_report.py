"""Builds results/single/single_elevator_results.pdf from single_results.json."""

import json
import os
import sys

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from make_report import make_image


def build_pdf(res, out):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
    story = []
    P = lambda t, st=body: story.append(Paragraph(t, st))

    o95, o99 = res["optimal"]["0.95"], res["optimal"]["0.99"]
    look, rnd = res["look"], res["random"]
    q = {g: [r for r in res["qlearning"] if r["gamma"] == g] for g in (0.95, 0.99)}
    qm = {g: (np.mean([r["test"]["people"] for r in v]), np.std([r["test"]["people"] for r in v]),
              np.mean([r["test"]["journey"] for r in v])) for g, v in q.items()}
    gap = lambda a, b: 100 * (a - b) / b
    S = res["settings"]

    story.append(Paragraph("E0: One Elevator as the Starting Point", ss["Title"]))
    P("Supplementary experiment. Code: <i>elevator/single_elevator.py</i> and <i>run_single.py</i>.", small)

    story.append(Paragraph("Setup", ss["Heading2"]))
    P("We use the single-elevator model of Yuan, Busoniu and Babuska (IFAC 2008): 5 floors, one car "
      "of capacity 4, time steps of 2 seconds, down-peak traffic (everyone travels to the ground "
      "floor, on average one passenger every 6.4 s, at most one person waiting per floor). "
      "The state is (call flags for floors 1-4, position, velocity, occupancy), 1200 states in total; "
      "actions are down, stop and up; the reward is minus the number of people waiting or riding. "
      "Because the model is small, the transition probabilities P(s' | s, a) can be written down "
      "exactly, so this is the one setting where we can compute the true optimal policy.")
    P("<b>Methods.</b> (1) <b>LOOK</b>: go up to the highest call, sweep down collecting people, "
      "unload at the ground floor, wait where you are when idle. (2) <b>Value iteration</b> on the exact "
      "model gives the optimal policy. (3) <b>Policy evaluation as an MRP</b>: fixing a policy turns "
      "the MDP into a Markov reward process, whose values solve V = (I - gamma P<sub>pi</sub>)<super>-1</super> "
      "R<sub>pi</sub>. (4) <b>Q-learning</b> from simulated experience only, without P, with "
      f"{S['episodes']:,} training episodes of {S['episode_steps']} steps starting from random states "
      f"(exploring starts), {S['seeds']} seeds. All policies are tested on the same {S['n_test']} "
      "trials of 500 seconds starting from an empty building.")

    story.append(Paragraph("Results", ss["Heading2"]))
    story.append(make_image(f"{out}/e0_policies.png", 14))
    rows = [["Policy", "People in system", "Journey time (s)", "vs optimal"]]
    best = o99["test"]["people"]
    rows.append(["Random", f"{rnd['people']:.2f}", f"{rnd['journey']:.1f}", f"{gap(rnd['people'], best):+.0f}%"])
    rows.append(["LOOK", f"{look['people']:.2f}", f"{look['journey']:.1f}", f"{gap(look['people'], best):+.1f}%"])
    for g, o in (("0.99", o99), ("0.95", o95)):
        rows.append([f"Optimal, value iteration (gamma {g})", f"{o['test']['people']:.2f}",
                     f"{o['test']['journey']:.1f}", f"{gap(o['test']['people'], best):+.1f}%"])
    for g in (0.95, 0.99):
        m, s, j = qm[g]
        rows.append([f"Q-learning (gamma {g})", f"{m:.2f} +/- {s:.2f}", f"{j:.1f}", f"{gap(m, best):+.0f}%"])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
                           ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
                           ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    story.append(t)
    P("Journey time = from arriving at a floor to reaching the ground floor. 'People in system' is "
      "minus the average reward, the quantity all policies are trying to minimise.", small)

    idle = o99["idle_moves"]
    P(f"<b>LOOK is almost optimal.</b> LOOK keeps {look['people']:.2f} people in the system on average "
      f"against {best:.2f} for the optimal policy ({gap(look['people'], best):.0f}% more), and its "
      f"journey time is {look['journey']:.1f} s against {o99['test']['journey']:.1f} s. In the "
      f"states the optimal policy actually visits, LOOK chooses the same action "
      f"{100 * res['look_agreement']:.0f}% of the time. The MRP calculation tells the same story: the "
      f"value of an empty building is {o99['v_empty']:.1f} under the optimal policy and "
      f"{o99['v_empty_look']:.1f} under LOOK. This confirms the professor's point that for a single "
      "elevator the classical method leaves little room for improvement.")
    P(f"<b>The one thing the optimal policy adds is parking.</b> When the building is empty, the "
      f"optimal car moves to floor 2, the middle floor (from floors 0 and 1 it goes up, from floors 3 "
      f"and 4 it comes down), so it is close to wherever the next call appears. LOOK simply waits "
      "wherever it finished. Anticipating demand instead of only reacting to it is the kind of "
      "behaviour RL can discover, but with one car it is worth only a few percent.")

    story.append(make_image(f"{out}/e0_learning.png", 10.5))
    P(f"<b>Learning from experience works, but the horizon matters.</b> With gamma = 0.95, "
      f"Q-learning reaches {qm[0.95][0]:.2f} people, within {gap(qm[0.95][0], o95['test']['people']):.0f}% "
      f"of the optimum for that gamma and close to LOOK. With gamma = 0.99 and the same training "
      f"budget it only reaches {qm[0.99][0]:.2f}, worse than LOOK, and its learning curve is unstable. With a "
      "horizon of about 100 steps, values have to be passed back through many more updates before "
      f"they settle (value iteration also needed {o99['iterations']} sweeps for gamma 0.99 against "
      f"{o95['iterations']} for 0.95). Starting episodes from random states was necessary: rare states, "
      "such as a full car with every floor calling, are almost never reached from an empty "
      "building, and without them the greedy policy could get stuck there.")

    story.append(Paragraph("Takeaway for the project", ss["Heading2"]))
    P("With one elevator there is a true optimum and LOOK is already within a few percent of it, so "
      "RL is hard to justify. Tabular methods also only work because the building is tiny: the state "
      "already has 1200 entries for 5 floors and grows exponentially with floors because every call "
      "button is part of the state. This is why the main project moves to several elevators, where "
      "the new decision (which car answers a call) has large, delayed consequences that heuristics "
      "handle poorly.")
    P("Note: we follow the paper's model but measure performance with journey times and the average "
      "number of people in the system. The paper's 'trial waiting time' is defined differently, so "
      "our numbers are not directly comparable with the values it reports.", small)

    SimpleDocTemplate(f"{out}/single_elevator_results.pdf", pagesize=A4, leftMargin=2 * cm,
                      rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="E0: One Elevator").build(story)
    print("wrote", f"{out}/single_elevator_results.pdf")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/single"
    with open(f"{out}/single_results.json") as fh:
        build_pdf(json.load(fh), out)
