#!/usr/bin/env python3
"""Aggregate results/summary.csv into results/results_table.md + plots/*.png."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(ROOT, "plots")
os.makedirs(P, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14, "figure.titlesize": 16})
COL = {"centroid_pid": "#64748b", "pid": "#f97316", "pure_pursuit": "#16a34a", "stanley_ff": "#2563eb"}
LBL = {"centroid_pid": "Baseline: threshold + centroid + PID", "pid": "Bird's-eye fit + PID",
       "pure_pursuit": "Bird's-eye fit + pure pursuit", "stanley_ff": "Bird's-eye fit + Stanley + curvature FF"}
SHORT = {"centroid_pid": "Centroid PID", "pid": "Fit + PID", "pure_pursuit": "Pure pursuit", "stanley_ff": "Stanley + FF"}
CTRLS = list(COL)
TRACKS = ["oval_chicane", "figure_eight", "technical"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
for f in ("Inter-Regular", "Inter-SemiBold"):
    p = f"/usr/share/fonts/opentype/inter/{f}.otf"
    if os.path.exists(p):
        matplotlib.font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = "Inter"

df = pd.read_csv(os.path.join(ROOT, "results", "summary.csv"))
df["completed"] = df["completed"].astype(bool)
comp = df[df.completed]

# ---- overall table per controller ----
rows = []
for c in CTRLS:
    d, dc = df[df.controller == c], comp[comp.controller == c]
    rows.append(dict(controller=SHORT[c], runs=len(d), completed=int(d.completed.sum()),
                     completion=f"{100*d.completed.mean():.0f}%",
                     clean=f"{int(d[d.condition=='clean'].completed.sum())}/{len(d[d.condition=='clean'])}",
                     hard=f"{int(d[d.condition=='hard'].completed.sum())}/{len(d[d.condition=='hard'])}",
                     cte_rms_cm=f"{dc.cte_rms_cm.mean():.2f}" if len(dc) else "-",
                     cte_max_cm=f"{dc.cte_max_cm.max():.1f}" if len(dc) else "-",
                     head_rms_deg=f"{dc.head_rms_deg.mean():.1f}" if len(dc) else "-",
                     lat_med_ms=f"{d.lat_med_ms.median():.2f}", lat_p95_ms=f"{d.lat_p95_ms.median():.2f}"))
overall = pd.DataFrame(rows)

# ---- completion by track/speed ----
piv = df.pivot_table(index=["track", "speed"], columns="controller", values="completed", aggfunc="sum").reindex(columns=CTRLS)
n = df.groupby(["track", "speed"]).seed.count().iloc[0] // len(CTRLS)

# ---- like-for-like: cases every controller finished on all seeds ----
ok_cases = df.groupby(["track", "condition", "speed"]).completed.all()
common = ok_cases[ok_cases].index
cm = df.set_index(["track", "condition", "speed"]).loc[common].reset_index()
fair = cm.groupby("controller")[["cte_rms_cm", "cte_max_cm", "head_rms_deg", "lap_time"]].mean().reindex(CTRLS)

with open(os.path.join(ROOT, "results", "results_table.md"), "w") as f:
    f.write(f"# Results ({len(df)} runs: 3 tracks x 2 lighting x 3 speeds x 4 controllers x 3 seeds)\n\n")
    f.write("Error stats are over completed laps only. Latency = per-frame perception time on one CPU core of my dev machine.\n\n")
    f.write(overall.to_markdown(index=False) + "\n\n")
    f.write("## Laps completed out of 6 per cell (clean + hard, 3 seeds each)\n\n")
    f.write(piv.astype(int).rename(columns=SHORT).to_markdown() + "\n\n")
    f.write(f"## Like-for-like: {len(common)} track/light/speed cases that all 4 finished on every seed\n\n")
    f.write(fair.rename(index=SHORT).round(2).to_markdown() + "\n")
print(overall.to_string(index=False))
print(piv)
print(fair.round(2))

# ---- plot 1: completion rate by track and speed ----
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
w = 0.2
for ax, tr in zip(axes, TRACKS):
    d = df[df.track == tr]
    sp = sorted(d.speed.unique())
    for k, c in enumerate(CTRLS):
        r = [100 * d[(d.speed == s) & (d.controller == c)].completed.mean() for s in sp]
        ax.bar(np.arange(len(sp)) + (k - 1.5) * w, r, w, color=COL[c], label=SHORT[c])
    ax.set_xticks(range(len(sp)), [f"{s:.2f} m/s" for s in sp])
    ax.set_title(tr.replace("_", " "))
    ax.set_ylim(0, 105)
axes[0].set_ylabel("laps completed (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=15, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Completion rate (clean + hard lighting, {n} seeds each = {2*n} laps per bar)")
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(os.path.join(P, "completion_rate.png"))

# ---- plot 2: cross-track RMS vs speed (completed laps) ----
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, tr in zip(axes, TRACKS):
    d = comp[comp.track == tr]
    for c in CTRLS:
        g = d[d.controller == c].groupby("speed").cte_rms_cm
        if len(g.mean()):
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().fillna(0).values, color=COL[c], marker="o",
                        capsize=3, label=SHORT[c])
    ax.set_title(tr.replace("_", " "))
    ax.set_xlabel("speed (m/s)")
axes[0].set_ylabel("cross-track RMS (cm)")
axes[0].legend(frameon=False, fontsize=13)
fig.suptitle("Cross-track error vs speed (completed laps, mean ± std over seeds and lighting)")
fig.tight_layout()
fig.savefig(os.path.join(P, "cte_rms_vs_speed.png"))

# ---- plot 3: CTE trace on the technical track, hard light, 0.19 m/s, seed 0 ----
fig, ax = plt.subplots(figsize=(12, 4))
for c in CTRLS:
    fn = os.path.join(ROOT, "logs", "technical", "hard", f"{c}_v0.19_s0.csv.gz")
    L = pd.read_csv(fn)
    ax.plot(L.t, L.cte * 100, color=COL[c], lw=1.2, label=SHORT[c])
    if abs(L.cte.iloc[-1]) > 0.119:
        ax.plot(L.t.iloc[-1], L.cte.iloc[-1] * 100, "x", color="#dc2626", ms=10, mew=2.5)
ax.axhline(12, color="#dc2626", ls="--", lw=1)
ax.axhline(-12, color="#dc2626", ls="--", lw=1)
ax.text(0.3, 12.4, "off-track limit ±12 cm", color="#dc2626", fontsize=13)
ax.set_xlabel("time (s)")
ax.set_ylabel("cross-track error (cm)")
ax.set_title("Technical track, hard lighting, 0.19 m/s, seed 0 (red x = left the line)")
ax.legend(frameon=False, ncol=4, fontsize=13, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(P, "cte_trace_technical_hard.png"))

# ---- plot 4: perception latency ----
fig, ax = plt.subplots(figsize=(7, 4))
lat = []
for c in ("centroid_pid", "stanley_ff"):
    v = []
    for tr in TRACKS:
        for cond in ("clean", "hard"):
            L = pd.read_csv(os.path.join(ROOT, "logs", tr, cond, f"{c}_v0.19_s0.csv.gz"), usecols=["lat_ms"])
            v.append(L.lat_ms.values)
    v = np.concatenate(v)
    lat.append(v)
ax.hist(lat[0], bins=np.linspace(0, 8, 81), color=COL["centroid_pid"], alpha=0.8, label=f"centroid (median {np.median(lat[0]):.2f} ms)")
ax.hist(lat[1], bins=np.linspace(0, 8, 81), color=COL["stanley_ff"], alpha=0.8,
        label=f"undistort + IPM + fit (median {np.median(lat[1]):.2f} ms, p95 {np.percentile(lat[1],95):.2f})")
ax.set_xlabel("perception time per frame (ms), 320x240, one CPU core")
ax.set_ylabel("frames")
ax.set_title("Per-frame perception latency")
ax.legend(frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(P, "perception_latency.png"))

# ---- plot 5: driven paths, technical track hard 0.19 s0 ----
sys.path.insert(0, ROOT)
from line_follower.track import Track
tr = Track("technical")
fig, ax = plt.subplots(figsize=(8, 6.4))
ax.plot(*np.vstack([tr.pts, tr.pts[:1]]).T, color="#111827", lw=3, alpha=0.25, label="track centreline")
for c in CTRLS:
    L = pd.read_csv(os.path.join(ROOT, "logs", "technical", "hard", f"{c}_v0.26_s0.csv.gz"))
    ax.plot(L.x, L.y, color=COL[c], lw=1.3, label=SHORT[c])
    if abs(L.cte.iloc[-1]) > 0.119:
        ax.plot(L.x.iloc[-1], L.y.iloc[-1], "x", color="#dc2626", ms=11, mew=2.5)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Driven paths, technical track, hard lighting, 0.26 m/s (red x = off track)")
ax.legend(frameon=False, fontsize=12, loc="upper center", ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(P, "paths_technical_fast.png"))

# ---- plot 6: perception accuracy ----
pa = json.load(open(os.path.join(ROOT, "results", "perception_accuracy.json")))
fig, ax = plt.subplots(figsize=(8, 4))
keys = list(pa)
x = np.arange(len(keys))
for k, (dist, colr) in enumerate([("20cm", "#0ea5e9"), ("30cm", "#6366f1"), ("45cm", "#a855f7")]):
    ax.bar(x + (k - 1) * 0.27, [pa[q]["rms_mm"][dist] for q in keys], 0.27, color=colr, label=f"{dist} ahead")
ax.set_xticks(x, [q.replace("_", " ").replace("/", "\n") for q in keys], fontsize=12)
ax.set_ylabel("lateral error RMS (mm)")
ax.set_title("Line fit vs true centreline (static poses, calibrated camera)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(P, "perception_accuracy.png"))
print("plots written")
