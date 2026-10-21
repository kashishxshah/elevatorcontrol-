"""
Turns results/results.json into figures (results/figures/*.png) and a
results PDF (results/elevator_rl_results.pdf).

    python make_report.py
    python make_report.py --results results/quick.json --pdf results/quick.pdf
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

from elevator.qlearning import state_space_size

METHODS = ["Random", "Nearest car", "ETA", "Q-learning"]
COLORS = {"Random": "#B0B0B0", "Nearest car": "#E69F00",
          "ETA": "#0072B2", "Q-learning": "#D55E00"}
PROFILE_NAMES = {"up_peak": "Up-peak", "down_peak": "Down-peak", "interfloor": "Interfloor"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


# ----------------------------------------------------------------------
# loading and aggregating
# ----------------------------------------------------------------------
class Results:
    def __init__(self, path):
        with open(path) as fh:
            data = json.load(fh)
        self.settings = data["settings"]
        self.exps = {k: [tuple(c) for c in v] for k, v in data["experiments"].items()}
        self.q = defaultdict(list)      # key -> list of per-seed records
        self.base = {}                  # (floors, cars, profile, method) -> metrics
        for r in data["records"]:
            c = r["cfg"]
            if r["method"] == "Q-learning":
                k = (c["floors"], c["cars"], c["profile"], c["lam"], c["gamma"])
                self.q[k].append(r)
            else:
                self.base[(c["floors"], c["cars"], c["profile"], r["method"])] = r["metrics"]

    def get(self, key, method, metric):
        """Return (mean, std over seeds) of a metric."""
        f, n, p, lam, g = key
        if method != "Q-learning":
            return self.base[(f, n, p, method)][metric], 0.0
        vals = [r["metrics"][metric] for r in self.q[key]]
        return float(np.mean(vals)), float(np.std(vals))

    def improvement(self, key, metric="mean"):
        """% reduction of Q-learning vs ETA (positive = Q-learning better)."""
        q, _ = self.get(key, "Q-learning", metric)
        e, _ = self.get(key, "ETA", metric)
        return 100 * (e - q) / e


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def fig_profiles(R, out):
    keys = R.exps["E1"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.2
    x = np.arange(len(keys))
    for i, m in enumerate(METHODS):
        vals = [R.get(k, m, "mean") for k in keys]
        ax.bar(x + (i - 1.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
               color=COLORS[m], label=m, capsize=2)
    ax.set_xticks(x, [PROFILE_NAMES[k[2]] for k in keys])
    ax.set_ylabel("Mean waiting time (ticks)")
    ax.set_title("E1: dispatchers on 10 floors, 3 cars")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_scaling(R, exp, idx, xlabel, title, out):
    keys = sorted(R.exps[exp], key=lambda k: k[idx])
    xs = [k[idx] for k in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for m in ["Nearest car", "ETA", "Q-learning"]:
        vals = [R.get(k, m, "mean") for k in keys]
        a1.errorbar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], marker="o",
                    color=COLORS[m], label=m, capsize=3)
    a1.set_xlabel(xlabel)
    a1.set_ylabel("Mean waiting time (ticks)")
    a1.set_yscale("log")
    a1.set_xticks(xs)
    a1.legend(frameon=False, fontsize=8)
    imp = [R.improvement(k) for k in keys]
    a2.bar([str(x) for x in xs], imp, color=[COLORS["Q-learning"] if v > 0 else "#999999" for v in imp])
    a2.axhline(0, color="black", lw=0.8)
    a2.set_xlabel(xlabel)
    a2.set_ylabel("Q-learning gain over ETA (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_tables(R, out):
    keys = sorted(R.exps["E3"], key=lambda k: k[1])
    cars = [k[1] for k in keys]
    full = [state_space_size(n, use_symmetry=False) for n in cars]
    sym = [state_space_size(n) for n in cars]
    seen = [R.get(k, "Q-learning", "states_visited")[0] for k in keys]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(cars, full, "s--", color="#999999", label="Possible states, cars labelled")
    ax.plot(cars, sym, "o-", color=COLORS["ETA"], label="Possible states, cars as types (used)")
    ax.plot(cars, seen, "^-", color=COLORS["Q-learning"], label="States visited in training")
    ax.set_yscale("log")
    ax.set_xticks(cars)
    ax.set_xlabel("Number of cars")
    ax.set_ylabel("Number of states")
    ax.set_title("Size of the Q-table as cars are added")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_curves(R, out):
    keys = sorted(R.exps["E3"], key=lambda k: k[1])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    shades = plt.cm.Oranges(np.linspace(0.45, 0.95, len(keys)))
    for k, col in zip(keys, shades):
        recs = R.q[k]
        eps = [c["episode"] for c in recs[0]["curve"]]
        eta, _ = R.get(k, "ETA", "mean")
        rel = np.mean([[c["mean"] / eta for c in r["curve"]] for r in recs], axis=0)
        ax.plot(eps, rel, marker="o", ms=3, color=col, label=f"{k[1]} cars")
    ax.axhline(1, color=COLORS["ETA"], ls="--", lw=1, label="ETA level")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean wait relative to ETA")
    ax.set_title("Learning curves (up-peak, 10 floors)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_fairness(R, out):
    keys = sorted(R.exps["E4"], key=lambda k: k[3])
    lams = [k[3] for k in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.3))
    for metric, style, label in [("max", "s-", "Max wait"), ("p95", "o-", "95th pct wait"),
                                 ("mean", "^-", "Mean wait")]:
        vals = [R.get(k, "Q-learning", metric) for k in keys]
        a1.errorbar(lams, [v[0] for v in vals], yerr=[v[1] for v in vals], fmt=style,
                    capsize=3, label=label)
    a1.set_xlabel("Fairness weight lambda")
    a1.set_ylabel("Waiting time (ticks)")
    a1.set_title("Waits vs lambda (mean +/- std over seeds)")
    a1.set_ylim(0, 75)
    a1.legend(frameon=False, fontsize=7, loc="upper center", ncol=3)

    jain = [R.get(k, "Q-learning", "jain") for k in keys]
    a2.errorbar(lams, [j[0] for j in jain], yerr=[j[1] for j in jain], fmt="o-",
                color=COLORS["Q-learning"], capsize=3, label="Q-learning")
    a2.axhline(R.get(keys[0], "ETA", "jain")[0], color=COLORS["ETA"], ls="--", label="ETA")
    a2.set_xlabel("Fairness weight lambda")
    a2.set_ylabel("Jain's fairness index")
    a2.set_title("Fairness index vs lambda")
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle("E4: fairness (up-peak, 20 floors, 3 cars)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_gamma(R, out):
    keys = sorted(R.exps["E5"], key=lambda k: k[4])
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    vals = [R.get(k, "Q-learning", "mean") for k in keys]
    labels = [f"{k[4]:g}\n(horizon ~{1 / (1 - k[4]):.0f} ticks)" for k in keys]
    ax.bar(labels, [v[0] for v in vals], yerr=[v[1] for v in vals],
           color=COLORS["Q-learning"], capsize=3, width=0.5)
    ax.axhline(R.get(keys[0], "ETA", "mean")[0], color=COLORS["ETA"], ls="--", label="ETA")
    ax.set_xlabel("Discount factor gamma")
    ax.set_ylabel("Mean waiting time (ticks)")
    ax.set_title("E5: effect of the discount factor")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------
def make_image(path, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)

def build_pdf(R, figdir, path):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
    h1, h2 = ss["Heading1"], ss["Heading2"]
    story = []

    def P(text, style=body):
        story.append(Paragraph(text, style))

    def fig(name, width=16):
        story.append(make_image(os.path.join(figdir, name), width))

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    def fmt(key, m, metric="mean"):
        v, s = R.get(key, m, metric)
        return f"{v:.1f} +/- {s:.1f}" if m == "Q-learning" else f"{v:.1f}"

    S = R.settings
    story.append(Paragraph("Multi-Elevator Dispatching with Tabular Q-Learning", ss["Title"]))
    P("Results of the experiments. Every number here is produced by <i>run_experiments.py</i> "
      "and this report by <i>make_report.py</i> from the saved results.", small)

    # --- setup -----------------------------------------------------------
    story.append(Paragraph("1. Setup", h1))
    P("<b>Simulator.</b> Discrete time: a car moves one floor per tick and a stop takes 2 ticks. "
      "Each car follows LOOK on its own stops. Passengers arrive as a Poisson process; "
      "the arrival rate is scaled with building height and number of cars so the load per car "
      f"is comparable (intensity {S['intensity']}). Car capacity is 8. "
      "Traffic profiles: up-peak (85% of trips from the lobby), down-peak (85% to the lobby) "
      "and interfloor (uniform).")
    P("<b>Decision problem.</b> Whenever a new hall call appears, a dispatcher chooses which car "
      "takes it. Baselines: <b>Random</b>; <b>Nearest car</b> (closest car by floor, ties broken at "
      "random); <b>ETA</b> (car with the smallest estimated time of arrival "
      "following LOOK, the standard industry rule).")
    P("<b>MDP and Q-learning.</b> State: whether the call is at the lobby, its direction, "
      "how many hall calls are open (4 levels), whether anyone has waited over 40 ticks, and "
      "each car's type = (ETA bucket, busyness bucket), listed as a sorted multiset because the "
      "cars are interchangeable. Action: which car type to send. Reward: minus the waiting cost "
      "paid until the next decision. Per tick each waiting passenger costs "
      "(1 - lambda) + lambda (2w + 1)/20, where w is their current wait; summed over a wait this "
      "gives the total wait when lambda = 0 and the total squared wait (scaled) when lambda = 1. "
      "Update: Q(s,a) &lt;- Q(s,a) + alpha [R + gamma<super>tau</super> max Q(s',a') - Q(s,a)], "
      "where tau is the number of ticks between decisions. Epsilon-greedy exploration decays from "
      "0.3 to 0.02; alpha = max(0.02, n<super>-0.6</super>) for the n-th visit of (s, a).")
    P(f"<b>Protocol.</b> {S['episodes']} training episodes of {S['ep_ticks']} ticks per run, "
      f"{S['seeds']} independent training seeds per setting (reported as mean +/- std across "
      f"seeds). Every method is tested on the same {S['tests']} unseen passenger streams; "
      f"passengers arriving in the first {S['warmup']} ticks are ignored. People still waiting at "
      "the end count with the wait so far. In test states never seen during training, the "
      "Q-learning dispatcher falls back to ETA; the share of decisions where this did not happen "
      "is reported as coverage. Waiting time is measured in ticks.")

    # --- E1 -------------------------------------------------------------
    story.append(Paragraph("2. Results", h1))
    story.append(Paragraph("E1. Traffic patterns (10 floors, 3 cars)", h2))
    fig("e1_profiles.png", 14)
    rows = [["Profile", "Random", "Nearest car", "ETA", "Q-learning", "Gain vs ETA"]]
    for k in R.exps["E1"]:
        rows.append([PROFILE_NAMES[k[2]]] + [fmt(k, m) for m in METHODS] + [f"{R.improvement(k):+.0f}%"])
    table(rows)
    P(TEXT.get("E1", ""))

    # --- E2 -------------------------------------------------------------
    story.append(Paragraph("E2. Building height (up-peak, 3 cars)", h2))
    fig("e2_floors.png", 16)
    rows = [["Floors", "Nearest car", "ETA", "Q-learning", "Gain vs ETA", "Q p95", "ETA p95"]]
    for k in sorted(R.exps["E2"], key=lambda k: k[0]):
        rows.append([k[0], fmt(k, "Nearest car"), fmt(k, "ETA"), fmt(k, "Q-learning"),
                     f"{R.improvement(k):+.0f}%", fmt(k, "Q-learning", "p95"), fmt(k, "ETA", "p95")])
    table(rows)
    P(TEXT.get("E2", ""))

    # --- E3 -------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("E3. Number of cars (up-peak, 10 floors)", h2))
    fig("e3_cars.png", 16)
    rows = [["Cars", "Nearest car", "ETA", "Q-learning", "Gain vs ETA", "Possible states",
             "Visited", "Coverage"]]
    for k in sorted(R.exps["E3"], key=lambda k: k[1]):
        rows.append([k[1], fmt(k, "Nearest car"), fmt(k, "ETA"), fmt(k, "Q-learning"),
                     f"{R.improvement(k):+.0f}%", f"{state_space_size(k[1]):,}",
                     f"{R.get(k, 'Q-learning', 'states_visited')[0]:,.0f}",
                     f"{100 * R.get(k, 'Q-learning', 'coverage')[0]:.1f}%"])
    table(rows)
    pair = Table([[make_image(os.path.join(figdir, "e3_tables.png"), 8.2),
                   make_image(os.path.join(figdir, "e3_curves.png"), 8.2)]])
    story.append(pair)
    P(TEXT.get("E3", ""))

    # --- E4 -------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("E4. Fairness vs efficiency (up-peak, 20 floors, 3 cars)", h2))
    fig("e4_fairness.png", 16)
    rows = [["lambda", "Mean wait", "95th pct", "Max wait", "Jain index"]]
    for k in sorted(R.exps["E4"], key=lambda k: k[3]):
        rows.append([f"{k[3]:g}"] + [fmt(k, "Q-learning", m) for m in ("mean", "p95", "max")]
                    + [f"{R.get(k, 'Q-learning', 'jain')[0]:.3f}"])
    k0 = R.exps["E4"][0]
    for m in ("ETA", "Nearest car"):
        rows.append([m] + [fmt(k0, m, x) for x in ("mean", "p95", "max")]
                    + [f"{R.get(k0, m, 'jain')[0]:.3f}"])
    table(rows)
    P("Jain's index is computed on (wait + 1); it equals 1 when everyone waits equally long. "
      "Note that it measures equality, not quality: Nearest car scores well only because "
      "almost everyone waits a long time.", small)
    P(TEXT.get("E4", ""))

    # --- E5 -------------------------------------------------------------
    story.append(Paragraph("E5. Discount factor and horizon", h2))
    fig("e5_gamma.png", 7.5)
    P(TEXT.get("E5", ""))

    # --- conclusions --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Conclusions", h1))
    for t in TEXT.get("conclusions", []):
        P("&bull; " + t)
    story.append(Paragraph("4. Limitations", h1))
    for t in TEXT.get("limitations", []):
        P("&bull; " + t)
    story.append(Paragraph("5. Further work", h1))
    for t in TEXT.get("further", []):
        P("&bull; " + t)

    SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Multi-Elevator Dispatching with Tabular Q-Learning").build(story)


TEXT = {}   # filled in by report_text.py once results are known

try:
    from report_text import make_text
except ImportError:
    make_text = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--figdir", default="results/figures")
    ap.add_argument("--pdf", default="results/elevator_rl_results.pdf")
    args = ap.parse_args()
    R = Results(args.results)
    os.makedirs(args.figdir, exist_ok=True)
    fd = args.figdir
    fig_profiles(R, f"{fd}/e1_profiles.png")
    fig_scaling(R, "E2", 0, "Floors", "E2: taller buildings (up-peak, 3 cars)", f"{fd}/e2_floors.png")
    fig_scaling(R, "E3", 1, "Cars", "E3: more cars (up-peak, 10 floors)", f"{fd}/e3_cars.png")
    fig_tables(R, f"{fd}/e3_tables.png")
    fig_curves(R, f"{fd}/e3_curves.png")
    fig_fairness(R, f"{fd}/e4_fairness.png")
    fig_gamma(R, f"{fd}/e5_gamma.png")
    if make_text:
        TEXT.update(make_text(R))
    build_pdf(R, fd, args.pdf)
    print("wrote", args.pdf)


if __name__ == "__main__":
    main()
