"""Step 4 - all final figures (F01-F15 and F04b).

Design rules applied to every figure:
  * the title states the takeaway; the subtitle defines marks, units, n and the test; the footer gives the source;
  * quantities are encoded by position or length; colour is categorical identity (one fixed colour per archetype)
    or a diverging scale whose midpoint is the all-country average;
  * bars start at zero, small multiples share axes, lines and groups are labelled directly;
  * derived / model-estimated values carry a dagger and an explanation in the footer.

Usage:  python scripts/figs.py [OUTPUT_DIR] [FIG_DIR] [DATA_DIR]   (defaults: outputs/  outputs/figs/  data/)
"""

import sys, json, warnings, numpy as np, pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
# --slides: write title-less variants for the presentation (the slide title carries the takeaway instead)
SLIDES = "--slides" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
O = ARGS[0] if len(ARGS) > 0 else str(ROOT / "outputs")
F = ARGS[1] if len(ARGS) > 1 else str(ROOT / "outputs" / ("figs_slides" if SLIDES else "figs"))
D = ARGS[2] if len(ARGS) > 2 else str(ROOT / "data")
TITLES = {}  # figure name -> takeaway title, written to titles.json in --slides mode
Path(F).mkdir(parents=True, exist_ok=True)
R = json.load(open(f"{O}/results.json"))
ORDER = R["order"]
N = ["Coal", "Gas", "Oil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other renewables"]
# One fixed colour per archetype in every figure (Okabe-Ito hues plus Tol "wine");
# checked with a colour-vision-deficiency validator: worst pair delta-E 11.0 under deuteranopia.
COL = {
    "Hydro-dominated": "#0072B2",
    "Diversified": "#009E73",
    "Gas-dominated": "#E69F00",
    "Oil-dependent": "#D55E00",
    "Coal-heavy": "#882255",
}
INK = "#222222"
MUTE = "#6b6b6b"
CTX = "#b8b8b8"
LIGHT = "#e6e6e6"
ACC = "#0072B2"
ACC2 = "#D55E00"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    }
)
SRC = "Data: Our World in Data energy dataset (Ember Yearly Electricity Data 2026; Energy Institute), CC BY 4.0."
import textwrap


def head(fig, title, sub, src=SRC, top=0.97):
    w, h = fig.get_size_inches()
    tl = textwrap.wrap(title, int(w * 8.4))
    sl = []
    for part in sub.split("\n"):
        sl += textwrap.wrap(part, int(w * 12.3))
    fl = textwrap.wrap(src, int(w * 16.6))
    if len(tl) > 1:
        print("  note: title wraps onto two lines:", title)
    y = 1 - 0.16 / h
    fig._takeaway = title
    if not SLIDES:
        fig.text(0.012, y, "\n".join(tl), fontsize=14, fontweight="bold", va="top", ha="left", color=INK, linespacing=1.15)
    y -= (0.29 * len(tl) + 0.08) / h
    fig.text(0.012, y, "\n".join(sl), fontsize=10.5, va="top", ha="left", color=MUTE, linespacing=1.3)
    fig.text(0.012, 0.07 / h, "\n".join(fl), fontsize=8, va="bottom", ha="left", color=MUTE, linespacing=1.3)


def fmt_signed(v, dec=0):
    v = round(float(v), dec) + 0.0
    return (
        "0"
        if v == 0 and dec == 0
        else f"{v:+.{dec}f}".replace("-", "−").replace("+0.0", "0.0") if v == 0 else f"{v:+.{dec}f}".replace("-", "−")
    )


def save(fig, name):
    path = f"{F}/{name}.png"
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)
    if SLIDES:
        # the title band is empty in this mode: trim the blank rows above the subtitle
        from PIL import Image

        im = Image.open(path).convert("RGB")
        arr = np.asarray(im)
        rows = np.where((arr < 250).any(axis=(1, 2)))[0]
        im.crop((0, max(rows[0] - 14, 0), im.width, im.height)).save(path)
        TITLES[name] = getattr(fig, "_takeaway", "")
    print("saved", name)


x = pd.read_csv(f"{O}/xs2023_labeled.csv")
panel = pd.read_csv(f"{O}/panel_clean.csv")
world = pd.read_csv(f"{O}/world.csv").set_index("year")
rng = np.random.default_rng(3)

# ---------------- F01 coverage ----------------
cov = pd.read_csv(f"{O}/coverage_by_year.csv", index_col=0).iloc[:, 0].loc[1985:2025]
fig, ax = plt.subplots(figsize=(11, 5.6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.15)
cols = [ACC if 2000 <= y <= 2024 else CTX for y in cov.index]
ax.bar(cov.index, cov.values, color=cols, width=0.8)
ax.set_ylim(0, 250)
ax.set_xlim(1984.3, 2028.5)
ax.set_ylabel("Countries with generation data")
ax.set_xlabel("Year")
ax.yaxis.grid(True, color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
ax.annotate(
    f"1985–1999: only {cov.loc[1985:1999].min()}–{cov[1999]} countries\n(Energy Institute series)",
    xy=(1992, cov[1999] + 4),
    ha="center",
    va="bottom",
    fontsize=10,
    color=MUTE,
)
ax.annotate(
    f"2000–2024: {cov.loc[2000:2023].min()}–{cov.loc[2000:2023].max()} countries ({cov[2024]} in 2024)\n= our analysis window",
    xy=(2012, cov[2012] + 4),
    ha="center",
    va="bottom",
    fontsize=10.5,
    color=ACC,
    fontweight="bold",
)
ax.annotate(
    f"2025: {cov[2025]}\n(partial year,\ndropped)", xy=(2024.6, cov[2025] + 4), ha="left", va="bottom", fontsize=9.5, color=MUTE
)
head(
    fig,
    "Coverage is complete only from 2000 to 2024, so that is the window we analyse",
    "Number of ISO-coded countries and territories reporting electricity generation, by year. Blue = years kept; grey = years dropped.",
)
save(fig, "F01_coverage")

# ---------------- F02 world trend ----------------
w = world.loc[2000:2024]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={"width_ratios": [1.15, 0.75]})
fig.subplots_adjust(left=0.06, right=0.865, top=0.76, bottom=0.13, wspace=1.0)
ser = [
    ("electricity_generation", "Electricity generated", INK, 2.4),
    ("greenhouse_gas_emissions", "Power-sector emissions †", ACC2, 2.4),
    ("carbon_intensity_elec", "Carbon intensity † (gCO₂e per kWh)", ACC, 2.4),
]
a1.axhline(100, color=CTX, lw=0.9)
for c, lab, colr, lw in ser:
    idx = 100 * w[c] / w.loc[2000, c]
    a1.plot(idx.index, idx.values, color=colr, lw=lw)
    v = idx.iloc[-1]
    a1.text(
        2024.5,
        v + {"electricity_generation": 6, "greenhouse_gas_emissions": -6}.get(c, 0),
        f"{lab.split(' (')[0]}\n{fmt_signed(v-100)}%",
        color=colr,
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
a1.set_xlim(2000, 2024)
a1.set_ylim(0, 220)
a1.set_xticks([2000, 2005, 2010, 2015, 2020, 2024])
a1.set_ylabel("Index (2000 = 100)")
a1.yaxis.grid(True, color=LIGHT, lw=0.8)
a1.set_axisbelow(True)
a1.set_title("a  World totals, indexed to 2000")
a1.text(2000.5, 95, "2000 level", color=MUTE, fontsize=9, va="top", ha="left")
s0, s1 = pd.Series(R["rq1"]["shares_2000"]), pd.Series(R["rq1"]["shares_2024"])
hl = {"Coal": "#882255", "Wind": "#009E73", "Solar": "#009E73", "Gas": "#E69F00"}


def spread(vals, mingap):
    order = np.argsort(vals)
    out = np.array(vals, float)
    for i in range(1, len(order)):
        if out[order[i]] - out[order[i - 1]] < mingap:
            out[order[i]] = out[order[i - 1]] + mingap
    return out


y0 = spread(s0[N].values, 2.7)
y1 = spread(s1[N].values, 2.7)
for i, nm in enumerate(N):
    c = hl.get(nm, CTX)
    lw = 2.4 if nm in hl else 1.4
    a2.plot([0, 1], [s0[nm], s1[nm]], color=c, lw=lw, marker="o", ms=5, zorder=3 if nm in hl else 2)
    tc = c if nm in hl else MUTE
    fwt = "bold" if nm in hl else "normal"
    a2.text(-0.05, y0[i], f"{nm}  {s0[nm]:.1f}%", ha="right", va="center", fontsize=9.5, color=tc, fontweight=fwt)
    a2.text(1.05, y1[i], f"{s1[nm]:.1f}%  {nm}", ha="left", va="center", fontsize=9.5, color=tc, fontweight=fwt)
a2.set_xlim(-0.02, 1.02)
a2.set_ylim(-1, 43)
a2.set_xticks([0, 1])
a2.set_xticklabels(["2000", "2024"], fontsize=11, color=INK)
a2.set_yticks([])
a2.spines["left"].set_visible(False)
a2.set_title("b  Share of world generation by source", x=-0.62)
head(
    fig,
    "Each kWh is 11% cleaner than in 2000, but the world makes twice as many of them",
    "World electricity, 2000–2024. Emissions = generation × carbon intensity, so a 102% rise in generation outweighs an 11% fall in intensity. Panel b: labels are nudged apart where values nearly coincide; the printed percentages are exact.",
    src=SRC + "  † Estimated by Ember from the generation mix using IPCC life-cycle emission factors (not measured).",
)
save(fig, "F02_world_trend")

# ---------------- F03 PCA ----------------
vr = np.array(R["pca"]["var_ratio"]) * 100
L = pd.read_csv(f"{O}/pca_loadings.csv", index_col=0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={"width_ratios": [0.8, 1.2]})
fig.subplots_adjust(left=0.06, right=0.97, top=0.78, bottom=0.12, wspace=0.22)
a1.bar(range(1, 9), vr[:8], color=[ACC] * 3 + [CTX] * 5, width=0.7)
for i, v in enumerate(vr[:8]):
    a1.text(
        i + 1, v + 1, f"{v:.1f}" if v >= 1 else f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK if i < 3 else MUTE
    )
a1.set_ylim(0, 55)
a1.set_xticks(range(1, 9))
a1.set_xlabel("Principal component")
a1.set_ylabel("Variance explained (%)")
a1.set_title("a  Scree plot")
a1.text(
    4.6,
    34,
    f"PC1–PC3 together:\n{vr[:3].sum():.1f}% of variance\n(PC1–PC2: {vr[:2].sum():.1f}%)",
    fontsize=10.5,
    color=ACC,
    fontweight="bold",
    va="center",
)
a2.scatter(x.PC1, x.PC2, s=22, color=CTX, edgecolor="white", lw=0.4, zorder=2)
sc = 95
for nm in ["Oil", "Gas", "Hydro", "Coal"]:
    lx, ly = L.loc[nm, "PC1"] * sc, L.loc[nm, "PC2"] * sc
    a2.annotate("", xy=(lx, ly), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=4)
    off = {"Oil": (9, 5), "Gas": (12, 6), "Hydro": (14, -3), "Coal": (-26, 7)}[nm]
    a2.text(lx + off[0], ly + off[1], nm, fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)
a2.set_xlabel(f"PC1 ({vr[0]:.1f}% of variance): oil  ↔  gas + hydro")
a2.set_ylabel(f"PC2 ({vr[1]:.1f}%): hydro  ↔  gas")
a2.set_aspect("equal", adjustable="datalim")
a2.axhline(0, color=LIGHT, lw=0.8, zorder=1)
a2.axvline(0, color=LIGHT, lw=0.8, zorder=1)
a2.set_title("b  PC1–PC2 scores (dot = power system) and loading directions")
head(
    fig,
    "Oil, gas and hydro span three-quarters of the variation between national electricity mixes",
    "PCA on the 9 generation shares of 211 power systems, 2023. Shares are all in % of generation, so they are centred but not z-scored. Coal loads on PC3.",
)
save(fig, "F03_pca")

# ---------------- F04 embeddings uncoloured vs coloured ----------------
fig, axs = plt.subplots(2, 2, figsize=(10.6, 7.6) if SLIDES else (12.5, 8.8))
fig.subplots_adjust(left=0.06, right=0.97, top=0.845, bottom=0.10, wspace=0.16, hspace=0.30)
LP = {
    "PCA": {
        "Gas-dominated": (-58, 68, "left"),
        "Hydro-dominated": (-58, -72, "left"),
        "Oil-dependent": (40, 22, "left"),
        "Coal-heavy": (8, -52, "left"),
        "Diversified": (16, 46, "left"),
    }
}
for r, (ex, ey, nm, xl, yl) in enumerate(
    [("PC1", "PC2", "PCA", f"PC1 ({vr[0]:.1f}%)", f"PC2 ({vr[1]:.1f}%)"), ("U1", "U2", "UMAP", "UMAP 1", "UMAP 2")]
):
    a_ = axs[r, 0]
    a_.scatter(x[ex], x[ey], s=20, color="#8a8a8a", edgecolor="white", lw=0.4)
    a_.set_title(f"{nm}: no colour")
    b_ = axs[r, 1]
    for g_ in ORDER:
        d = x[x.archetype == g_]
        b_.scatter(d[ex], d[ey], s=22, color=COL[g_], edgecolor="white", lw=0.4)
    b_.set_title(f"{nm}: coloured by k-means archetype (n)")
    for q in (a_, b_):
        q.set_xlabel(xl)
        q.set_ylabel(yl)
        if nm == "UMAP":
            q.set_xticks([])
            q.set_yticks([])
    if nm == "PCA":
        for q in (a_, b_):
            q.set_xlim(-62, 112)
            q.set_ylim(-82, 82)
        for g_ in ORDER:
            d = x[x.archetype == g_]
            cx, cy = d[ex].median(), d[ey].median()
            lx, ly, ha = LP["PCA"][g_]
            b_.annotate(
                f"{g_} ({len(d)})",
                xy=(cx, cy),
                xytext=(lx, ly),
                fontsize=9.5,
                fontweight="bold",
                color=INK,
                ha=ha,
                va="center",
                arrowprops=dict(arrowstyle="-", color=COL[g_], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL[g_], lw=1.4),
            )
        b_.text(
            110,
            80,
            "Coal-heavy and Diversified overlap here;\nthey separate on PC3 (13.9%)",
            fontsize=9,
            color=MUTE,
            ha="right",
            va="top",
        )
    else:
        xr = x[ex].max() - x[ex].min()
        yr = x[ey].max() - x[ey].min()
        for q in (a_, b_):
            q.set_xlim(x[ex].min() - 0.05 * xr, x[ex].max() + 0.30 * xr)
            q.set_ylim(x[ey].min() - 0.08 * yr, x[ey].max() + 0.10 * yr)
        UL = {
            "Gas-dominated": (0.28, 0.05),
            "Diversified": (0.38, 0.03),
            "Coal-heavy": (-0.02, -0.34),
            "Hydro-dominated": (0.16, -0.42),
            "Oil-dependent": (0.42, -0.20),
        }
        for g_ in ORDER:
            d = x[x.archetype == g_]
            cx, cy = d[ex].median(), d[ey].median()
            ox, oy = UL[g_]
            b_.annotate(
                f"{g_} ({len(d)})",
                xy=(cx, cy),
                xytext=(cx + ox * xr, cy + oy * yr),
                fontsize=9.5,
                fontweight="bold",
                color=INK,
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", color=COL[g_], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL[g_], lw=1.4),
            )
        far = x[x.Oil >= 95]
        b_.annotate(
            f"{len(far)} systems that are\n95–100% oil",
            xy=(far[ex].median(), far[ey].median()),
            xytext=(far[ex].median() - 0.26 * xr, far[ey].median() + 0.02 * yr),
            fontsize=9,
            color=MUTE,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-", color=CTX, lw=1),
        )
head(
    fig,
    "The grouping is visible before any colour is added, so the archetypes are not painted on",
    "Same 211 power systems in each panel. Top: PCA scores. Bottom: UMAP (n_neighbors = 15, min_dist = 0.1, Euclidean distance, PCA initialisation, seed 0).",
    src="UMAP axes have no units; distances between UMAP groups and the sizes of those groups are not interpretable. Clusters were fitted in the original 9-dimensional share space, not on these 2-D views. "
    + SRC,
)
save(fig, "F04_embeddings")

# ---------------- F04b UMAP stability grid ----------------
G = np.load(f"{O}/umap_grid.npy")
SS = [(15, 0.1, 0), (15, 0.1, 1), (15, 0.1, 2), (10, 0.1, 0), (30, 0.1, 0), (15, 0.3, 0)]
us = pd.read_csv(f"{O}/umap_stability.csv")
fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.6))
fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.14, wspace=0.08, hspace=0.30)
for a_, E, (nn, md, seed) in zip(axs.ravel(), G, SS):
    for g_ in ORDER:
        m_ = (x.archetype == g_).values
        a_.scatter(E[m_, 0], E[m_, 1], s=13, color=COL[g_], edgecolor="white", lw=0.3)
    tr = us[(us.n_neighbors == nn) & (us.min_dist == md) & (us.seed == seed)].trust.iloc[0]
    a_.set_title(f"n_neighbors = {nn}, min_dist = {md}, seed {seed}", fontsize=10.5)
    a_.set_xlabel(f"trustworthiness {tr:.3f}", fontsize=9, color=MUTE, loc="left")
    a_.set_xticks([])
    a_.set_yticks([])
    for sp_ in a_.spines.values():
        sp_.set_color(LIGHT)
hs = [Line2D([], [], marker="o", ls="", color=COL[g_], ms=7, label=g_) for g_ in ORDER]
fig.legend(
    handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=5, fontsize=9.5, handletextpad=0.2, columnspacing=1.6
)
head(
    fig,
    "UMAP layouts change with seed and settings, but the same five groups stay together",
    f"Six UMAP runs of the same 211 systems (top row: three seeds; bottom row: other hyper-parameters). Trustworthiness (k = 10) ranges {R['umap']['trust_min']:.3f}–{R['umap']['trust_max']:.3f} across 20 runs; PCA's first two components score {R['umap']['trust_pca2']:.3f}.",
    src="Positions, orientation and gaps differ between runs, so only group membership is read from UMAP. Colours are the k-means archetypes fitted in the original share space. "
    + SRC,
)
save(fig, "F04b_umap_stability")

# ---------------- F05 choosing k ----------------
ks = pd.read_csv(f"{O}/k_selection.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.15, wspace=0.24)
a1.plot(ks.k, ks.silhouette, color=MUTE, lw=1.6, marker="o", ms=6, mfc="white", mec=MUTE)
r5 = ks[ks.k == 5].iloc[0]
a1.plot([5], [r5.silhouette], marker="o", ms=10, color=ACC)
for r in ks.itertuples():
    a1.text(
        r.k,
        r.silhouette + 0.014,
        f"{r.silhouette:.3f}" if r.k in (4, 5) else f"{r.silhouette:.2f}",
        ha="center",
        fontsize=9,
        color=ACC if r.k == 5 else MUTE,
        fontweight="bold" if r.k == 5 else "normal",
    )
a1.set_ylim(0, 0.6)
a1.set_xticks(ks.k)
a1.set_xlabel("Number of clusters k")
a1.set_ylabel("Mean silhouette width")
a1.set_title("a  Separation: higher is better")
a1.yaxis.grid(True, color=LIGHT, lw=0.8)
a1.set_axisbelow(True)
a2.fill_between(ks.k, ks.ari_p05, ks.ari_p95, color=LIGHT, zorder=1)
a2.plot(ks.k, ks.ari_mean, color=MUTE, lw=1.6, marker="o", ms=6, mfc="white", mec=MUTE, zorder=2)
a2.plot([5], [r5.ari_mean], marker="o", ms=10, color=ACC, zorder=3)
for r in ks.itertuples():
    a2.text(
        r.k,
        r.ari_p95 + 0.015,
        f"{r.ari_mean:.2f}",
        ha="center",
        fontsize=9,
        color=ACC if r.k == 5 else MUTE,
        fontweight="bold" if r.k == 5 else "normal",
    )
a2.set_ylim(0, 1.08)
a2.set_xticks(ks.k)
a2.set_xlabel("Number of clusters k")
a2.set_ylabel("Adjusted Rand Index vs full-data solution")
a2.set_title("b  Stability over 200 bootstrap resamples")
a2.yaxis.grid(True, color=LIGHT, lw=0.8)
a2.set_axisbelow(True)
head(
    fig,
    "k = 5 ties k = 4 on separation and stays stable; from k = 6 both scores fall",
    "k-means on 9 generation shares, 211 power systems, 2023. Blue = chosen k. Band in panel b = 5th–95th percentile. k = 4 merges coal-heavy with diversified systems; k = 6 splits the oil group.",
)
save(fig, "F05_choose_k")

# ---------------- F06 profile heatmap ----------------
T = pd.read_csv(f"{O}/tests_external.csv")
DS = pd.read_csv(f"{O}/descriptive_separation_inputs.csv")
mean = pd.read_csv(f"{O}/cluster_means.csv", index_col=0).loc[ORDER]
z = pd.read_csv(f"{O}/cluster_zprofile.csv", index_col=0).loc[ORDER]
colorder = DS.col.tolist()
eff = dict(zip(DS.col, DS.eta2_H))
fig, ax = plt.subplots(figsize=(12.5, 5.9))
fig.subplots_adjust(left=0.20, right=0.90, top=0.70, bottom=0.09)
Zm = z[colorder].values
vmax = 3
im = ax.imshow(Zm, cmap="PuOr_r", vmin=-vmax, vmax=vmax, aspect="auto")
for i, g in enumerate(ORDER):
    for j, c in enumerate(colorder):
        v = mean.loc[g, c]
        zz = Zm[i, j]
        ax.text(
            j,
            i,
            f"{v:.0f}%" if v >= 0.5 else "<1%",
            ha="center",
            va="center",
            fontsize=11,
            color="white" if abs(zz) > 1.7 else INK,
            fontweight="bold" if zz > 1 else "normal",
        )
ax.set_xticks(range(len(colorder)))
ax.set_xticklabels([f"{c}\nη² = {eff[c]:.2f}" for c in colorder], fontsize=9.5, color=INK)
ax.xaxis.tick_top()
ax.tick_params(length=0)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([f"{g}  (n = {R['cluster']['sizes'][g]})" for g in ORDER], fontsize=10.5, color=INK)
ax.tick_params(axis="y", pad=16)
for i, g in enumerate(ORDER):
    ax.add_patch(plt.Rectangle((-0.5 - 0.085, i - 0.47), 0.06, 0.94, color=COL[g], clip_on=False, transform=ax.transData))
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(colorder), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("z-score vs. all 211 systems\n(0 = average share)", fontsize=9)
cb.outline.set_visible(False)
cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cb.ax.set_yticklabels(["≤ −3", "−2", "−1", "0", "+1", "+2", "≥ +3"], fontsize=8.5)
head(
    fig,
    "Four archetypes lean on one source; the fifth spreads across nuclear, wind, solar and bioenergy",
    "Cell text = mean share of generation in the archetype (2023). Colour = how far that mean sits from the all-country average, in standard deviations.\nColumns ordered by descriptive separation (η² from a Kruskal–Wallis rank decomposition). These shares built the clusters, so η² only orders the columns: it is not a test and no p-values are reported.",
)
save(fig, "F06_profiles")

# ---------------- F07 counts vs generation ----------------
S = pd.read_csv(f"{O}/cluster_summary.csv", index_col=0).loc[ORDER]
fig, axs = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=True)
fig.subplots_adjust(left=0.15, right=0.97, top=0.74, bottom=0.12, wspace=0.10)
ypos = np.arange(len(ORDER))[::-1]
for a, (col, ttl, fmt) in zip(
    axs,
    [
        ("n_pct", "a  Share of the 211 power systems", "{:.0f}%"),
        ("pop_share", "b  Share of population covered", "{:.0f}%"),
        ("gen_share", "c  Share of electricity generated", "{:.1f}%"),
    ],
):
    vals = (100 * S.n / S.n.sum()) if col == "n_pct" else S[col]
    a.barh(ypos, vals.values, color=[COL[g] for g in ORDER], height=0.62)
    for yy, v, g in zip(ypos, vals.values, ORDER):
        t = fmt.format(v) + (f"  ({int(S.loc[g,'n'])} systems)" if col == "n_pct" else "")
        a.text(v + 1, yy, t, va="center", ha="left", fontsize=10, color=INK)
    a.set_xlim(0, 62)
    a.set_title(ttl)
    a.set_xticks([0, 25, 50])
    a.set_xticklabels(["0", "25%", "50%"])
    a.xaxis.grid(True, color=LIGHT, lw=0.8)
    a.set_axisbelow(True)
    a.tick_params(axis="y", length=0)
axs[0].set_yticks(ypos)
axs[0].set_yticklabels(ORDER, fontsize=10.5, color=INK)
head(
    fig,
    "61 oil-dependent systems make 0.4% of world electricity; 25 coal-heavy ones make half",
    "Each archetype's share of power systems, of people and of electricity generated, 2023. All bars start at zero and share one scale. Counting countries alone would overstate oil and understate coal.",
)
save(fig, "F07_weight")

# ---------------- F08 map ----------------
import geopandas as gpd

g = gpd.read_file(f"{D}/ne/ne_110m_admin_0_countries.geojson")[["ADM0_A3", "ISO_A3_EH", "NAME", "geometry"]]
g = g[g.NAME != "Antarctica"]
g["iso"] = g.ISO_A3_EH.where(g.ISO_A3_EH != "-99", g.ADM0_A3).replace(
    {"KOS": "OWID_KOS", "SDS": "SSD", "PSX": "PSE", "SAH": "ESH"}
)
g = g.merge(x[["iso_code", "archetype"]], left_on="iso", right_on="iso_code", how="left").to_crs("EPSG:8857")
print("map: unmatched polygons:", g[g.archetype.isna()].NAME.tolist())
print(
    "map: systems not drawn:",
    len(set(x.iso_code) - set(g.iso)),
    "of which oil:",
    (x[~x.iso_code.isin(g.iso)].archetype == "Oil-dependent").sum(),
)
nd = x[~x.iso_code.isin(g.iso)]
fig, ax = plt.subplots(figsize=(12.5, 6.6))
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.09)
g[g.archetype.isna()].plot(ax=ax, color="#f0f0f0", edgecolor="white", lw=0.3)
for a_ in ORDER:
    g[g.archetype == a_].plot(ax=ax, color=COL[a_], edgecolor="white", lw=0.3)
ax.set_axis_off()
handles = [Patch(fc=COL[a_], label=f"{a_} ({R['cluster']['sizes'][a_]})") for a_ in ORDER] + [
    Patch(fc="#f0f0f0", ec="#cccccc", label="No data")
]
ax.legend(
    handles=handles,
    loc="lower left",
    bbox_to_anchor=(0.0, 0.02),
    fontsize=9.5,
    title="Archetype (systems)",
    title_fontsize=9.5,
    alignment="left",
)
head(
    fig,
    "Archetypes follow geography: South American and African hydro, Gulf gas, Asian coal",
    "Electricity-mix archetype of each country, 2023 (k-means, k = 5). Equal Earth projection, so areas are comparable; colour is identity only.",
    src=f"{len(nd)} of 211 systems are too small to draw at this scale ({(nd.archetype=='Oil-dependent').sum()} of them oil-dependent islands), so the map under-shows the oil group. Area shows land, not electricity. "
    + SRC
    + " Boundaries: Natural Earth.",
)
save(fig, "F08_map")


# ---------------- helpers: box + jitter ----------------
def boxjit(ax, groups, vals_by, colors=None, point_colors=None, width=0.5, notch=True, logspace=False):
    """Notched box (median, IQR, 1.5 x IQR whiskers) with every observation jittered behind it.
    Notch = median +/- 1.57 x IQR / sqrt(n), the usual approximate 95% CI of the median (McGill et al. 1978),
    i.e. what matplotlib draws with notch=True. With logspace=True the notch is computed on log10 values.
    Returns {group: (notch_low, notch_high, q1, q3)} so callers can report notches that pass a quartile."""
    out = {}
    for i, gname in enumerate(groups):
        v = np.asarray(vals_by[gname], float)
        q1, m, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        lo = v[v >= q1 - 1.5 * iqr].min()
        hi = v[v <= q3 + 1.5 * iqr].max()
        if logspace:
            lq1, lm, lq3 = np.percentile(np.log10(v), [25, 50, 75])
            h = 1.57 * (lq3 - lq1) / np.sqrt(len(v))
            nlo, nhi = 10 ** (lm - h), 10 ** (lm + h)
        else:
            h = 1.57 * iqr / np.sqrt(len(v))
            nlo, nhi = m - h, m + h
        out[gname] = (nlo, nhi, q1, q3)
        w = width / 2
        wn = width / 4 if notch else w
        if notch:
            pts = [
                (i - w, q1),
                (i + w, q1),
                (i + w, nlo),
                (i + wn, m),
                (i + w, nhi),
                (i + w, q3),
                (i - w, q3),
                (i - w, nhi),
                (i - wn, m),
                (i - w, nlo),
            ]
        else:
            pts = [(i - w, q1), (i + w, q1), (i + w, q3), (i - w, q3)]
        ax.add_patch(plt.Polygon(pts, closed=True, fc="none", ec="#555555", lw=1.1, zorder=3, joinstyle="miter"))
        ax.plot([i - wn, i + wn], [m, m], color=INK, lw=2.4, zorder=4, solid_capstyle="butt")
        ax.plot([i, i], [lo, q1], color="#555555", lw=1, zorder=3)
        ax.plot([i, i], [q3, hi], color="#555555", lw=1, zorder=3)
        jx = i + rng.uniform(-w + 0.04, w - 0.04, len(v))
        pc = point_colors[gname] if point_colors is not None else (colors[gname] if colors else CTX)
        ax.scatter(jx, v, s=15, color=pc, alpha=0.75, edgecolor="none", zorder=2)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)))
    return out


# ---------------- F09 carbon intensity by archetype ----------------
fig, ax = plt.subplots(figsize=(11.5, 6.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.17)
vb = {g_: x[x.archetype == g_].carbon_intensity_elec.values for g_ in ORDER}
nt = boxjit(ax, ORDER, vb, colors=COL)
past = [g_ for g_, (a_, b_, q1_, q3_) in nt.items() if a_ < q1_ or b_ > q3_]
print("F09 notches:", {g_: (round(a_), round(b_)) for g_, (a_, b_, _, _) in nt.items()}, "| past a quartile:", past)
for i, g_ in enumerate(ORDER):
    ax.text(
        i + 0.29,
        np.median(vb[g_]),
        f"median {np.median(vb[g_]):.0f}",
        va="center",
        ha="left",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )
ax.set_xticklabels([f"{g_}\nn = {len(vb[g_])}" for g_ in ORDER], fontsize=10.5, color=INK)
ax.set_ylim(0, 1400)
ax.set_ylabel("Carbon intensity † (gCO₂e per kWh)")
ax.yaxis.grid(True, color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
tt = T[T.col == "carbon_intensity_elec"].iloc[0]
ax.text(
    -0.5,
    1340,
    "Pairwise Dunn tests, Holm-adjusted:\nhydro vs diversified p = 1.00\ngas vs oil vs coal: all p = 1.00\nevery low-carbon vs fossil pair: p < 0.001",
    fontsize=9.5,
    color=MUTE,
    ha="left",
    va="top",
    linespacing=1.4,
)
ax.text(2.33, 1306, "Turkmenistan", fontsize=8.5, color=MUTE, va="center")
ax.text(2.33, 1113, "Uzbekistan", fontsize=8.5, color=MUTE, va="center")
ax.text(3.33, 1000, "3 micro-systems\n(rounding artefact)", fontsize=8.5, color=MUTE, va="center")
head(
    fig,
    "Two tiers: hydro and diversified systems emit a quarter as much per kWh as fossil ones",
    "Carbon intensity of electricity by archetype, 2023. Box = median and interquartile range; notch = 95% CI of the median (non-overlapping notches suggest different medians); whiskers = 1.5 × IQR; dots = power systems.\n"
    + f"Kruskal–Wallis H({int(tt.df)}) = {tt.stat:.1f}, n = {int(tt.n)}, q < 0.001 (Benjamini–Hochberg across the 8 external variables), η² = {tt.effect:.2f}.",
    src=SRC
    + "  † Derived variable: estimated from the generation mix with IPCC emission factors, so this difference is expected from the clustering inputs and is not independent validation.",
)
save(fig, "F09_ci_by_archetype")

# ---------------- F10 income ----------------
IO = ["Low income", "Lower middle income", "Upper middle income", "High income"]
d = x.dropna(subset=["income_group"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.4))
fig.subplots_adjust(left=0.07, right=0.975, top=0.74, bottom=0.25, wspace=0.24)
vb = {g_: d[d.income_group == g_].carbon_intensity_elec.values for g_ in IO}
pc = {g_: [COL[a_] for a_ in d[d.income_group == g_].archetype] for g_ in IO}
nt1 = boxjit(a1, IO, vb, point_colors=pc)
print("F10a notches past a quartile:", [g_ for g_, (a_, b_, q1_, q3_) in nt1.items() if a_ < q1_ or b_ > q3_])
for i, g_ in enumerate(IO):
    a1.text(i + 0.29, np.median(vb[g_]), f"{np.median(vb[g_]):.0f}", va="center", ha="left", fontsize=9.5, fontweight="bold")
a1.set_ylim(0, 1400)
a1.set_ylabel("Carbon intensity † (gCO₂e per kWh)")
a1.set_xticklabels([f"{g_.replace(' income','')}\nn = {len(vb[g_])}" for g_ in IO], fontsize=10, color=INK)
q = R["rq3"]
a1.set_title(f"a  Cleaner? No:  H(3) = {q['kw_H']:.1f}, p = {q['kw_p']:.3f}, η² = {q['eta2_H']:.2f}")
vk = {g_: d[d.income_group == g_].per_capita_electricity.values for g_ in IO}
nt2 = boxjit(a2, IO, vk, logspace=True)
print("F10b notches past a quartile:", [g_ for g_, (a_, b_, q1_, q3_) in nt2.items() if a_ < q1_ or b_ > q3_])
a2.set_yscale("log")
a2.set_ylim(10, 80000)
a2.set_yticks([10, 100, 1000, 10000])
a2.set_yticklabels(["10", "100", "1,000", "10,000"])
a2.set_ylabel("kWh generated per person (log scale)")
for i, g_ in enumerate(IO):
    a2.text(i + 0.29, np.median(vk[g_]), f"{np.median(vk[g_]):,.0f}", va="center", ha="left", fontsize=9.5, fontweight="bold")
a2.set_xticklabels([f"{g_.replace(' income','')}\nn = {len(vk[g_])}" for g_ in IO], fontsize=10, color=INK)
kk = q["kw_kwh"]
a2.set_title(f"b  More? Yes:  H(3) = {kk['H']:.1f}, p < 0.001, η² = {kk['eta2_H']:.2f}")
for a in (a1, a2):
    a.yaxis.grid(True, color=LIGHT, lw=0.8)
    a.set_axisbelow(True)
    a.set_xlabel("World Bank income group")
hs = [Line2D([], [], marker="o", ls="", color=COL[a_], ms=7, label=a_) for a_ in ORDER]
fig.legend(
    handles=hs,
    loc="lower left",
    bbox_to_anchor=(0.065, 0.065),
    ncol=5,
    fontsize=9.5,
    handletextpad=0.2,
    columnspacing=1.4,
    title="Dot colour in panel a = archetype",
    title_fontsize=9.5,
    alignment="left",
)
head(
    fig,
    "Income buys more electricity, not cleaner electricity",
    "202 power systems with a World Bank income group, 2023. Box = median and IQR; notch = 95% CI of the median (panel b: on log values); whiskers = 1.5 × IQR; bold numbers = medians. Kruskal–Wallis tests of the pre-specified hypothesis, unadjusted.\n"
    + f"Spearman GDP per capita vs carbon intensity: ρ = {fmt_signed(q['spearman_gdp_ci']['rho'],2)} (95% CI {fmt_signed(q['spearman_gdp_ci']['ci_lo'],2)} to {fmt_signed(q['spearman_gdp_ci']['ci_hi'],2)}, n = {q['spearman_gdp_ci']['n']}); vs kWh per person: ρ = {q['spearman_gdp_kwh']['rho']:.2f} ({q['spearman_gdp_kwh']['ci_lo']:.2f} to {q['spearman_gdp_kwh']['ci_hi']:.2f}).",
    src=SRC
    + " Income groups: World Bank (2020 vintage). GDP: Maddison Project 2023 (2022 values).  † Estimated with IPCC emission factors.",
)
save(fig, "F10_income")

# ---------------- F11 effect sizes (external variables only) ----------------
# Three groups, top to bottom: (1) carbon intensity, which is DERIVED from the mix and therefore expected to differ;
# (2) independent numeric variables; (3) independent categorical variables. BH correction across these 8 tests.
dv = T[T.role != "independent"]
kw = T[(T.role == "independent") & (T.test == "Kruskal-Wallis")].sort_values("effect")
ch = T[(T.role == "independent") & (T.test != "Kruskal-Wallis")].sort_values("effect")
fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(11.5, 6.4), gridspec_kw={"height_ratios": [1.15, len(kw), len(ch)]}, sharex=True)
fig.subplots_adjust(left=0.27, right=0.80, top=0.76, bottom=0.15, hspace=0.50)


def fq(v):
    return "q < 0.001" if v < 0.001 else f"q = {v:.3f}"


for a, dd, ttl, derived in [
    (a0, dv, "Derived from the clustering inputs: expected, not validation (η²)", True),
    (a1, kw, "Independent numeric variables: Kruskal–Wallis η²", False),
    (a2, ch, "Independent categorical variables: χ² (Monte-Carlo), Cramér's V", False),
]:
    yy = np.arange(len(dd))
    e = dd.effect.clip(lower=0).values
    sig = dd.q_BH.values < 0.05
    lc = [MUTE if derived else (ACC if s_ else CTX) for s_ in sig]
    a.hlines(yy, 0, e, color=lc, lw=2.2, linestyles=(0, (4, 2)) if derived else "solid")
    a.scatter(
        e,
        yy,
        s=70,
        color=["white" if (derived or not s_) else ACC for s_ in sig],
        edgecolor=[MUTE if (derived or not s_) else ACC for s_ in sig],
        lw=1.6,
        marker="D" if derived else "o",
        zorder=3,
    )
    a.set_yticks(yy)
    a.set_yticklabels(dd.variable, fontsize=10, color=MUTE if derived else INK)
    a.set_ylim(-0.6, len(dd) - 0.4)
    a.set_title(ttl, fontsize=10.5, color=MUTE if derived else INK)
    for y_, r in zip(yy, dd.itertuples()):
        a.text(
            1.03,
            y_,
            f"{max(r.effect,0):.2f}   {fq(r.q_BH)}   n = {r.n}",
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK if (r.q_BH < 0.05 and not derived) else MUTE,
            transform=a.get_yaxis_transform(),
        )
    a.set_xlim(0, 1)
    a.xaxis.grid(True, color=LIGHT, lw=0.8)
    a.set_axisbelow(True)
    a.tick_params(axis="y", length=0)
for t_ in (0.01, 0.06, 0.14):
    a1.axvline(t_, color=CTX, lw=0.8, ls=(0, (2, 2)), zorder=0)
a1.text(
    0.148,
    0.0,
    "dotted lines: η² = 0.01 small,\n0.06 medium, 0.14 large (Cohen)",
    fontsize=8.5,
    color=MUTE,
    va="center",
    ha="left",
)
a2.set_xlabel("Effect size (0 = no association, 1 = perfect)")
head(
    fig,
    "Archetypes differ in system size, continent and wealth, but not in import dependence",
    "Association between archetype and the 8 variables that were NOT used for clustering, 2023. Filled circle = q < 0.05 after Benjamini–Hochberg across these 8 tests; hollow = not significant; grey diamond = derived variable.",
    src=SRC
    + "  † Carbon intensity is computed from the mix with IPCC emission factors, so it is reported for completeness only. A negative η² estimate is shown as 0. Associations, not causes.",
)
save(fig, "F11_effects")

# ---------------- F12 surrogates ----------------
PI = pd.read_csv(f"{O}/rf_perm_importance.csv")
CO = pd.read_csv(f"{O}/logit_coefs.csv", index_col=0).loc[ORDER]
SV = pd.read_csv(f"{O}/surrogates_cv.csv").set_index("model")
chain = [(c["feature"], c["threshold"], c["leaf"], c["n_train"]) for c in R["tree"]["chain"]]
final = (R["tree"]["else_leaf"], R["tree"]["else_n_train"])
fig = plt.figure(figsize=(13, 6.8))
head(
    fig,
    "Four yes/no questions reproduce the five archetypes with 95% balanced accuracy on unseen systems",
    f"Surrogate models predicting the k-means archetype from the 9 shares. 10 × repeated stratified 5-fold CV balanced accuracy: tree {SV.loc['Decision tree (5 leaves)','bal_acc']:.2f}, random forest {SV.loc['Random forest (500 trees)','bal_acc']:.2f}, logistic {SV.loc['Multinomial logistic (L2, C=1, z-scored)','bal_acc']:.2f} (majority-class baseline 0.20).",
    src="Tree and importances fitted on a 70% training split (n = 147) and evaluated on the 30% hold-out (n = 64). These models describe the cluster boundaries; they do not explain why a country has its mix. Divers. = Diversified. "
    + SRC,
)
a0 = fig.add_axes([0.01, 0.10, 0.40, 0.68])
a0.set_xlim(0, 10)
a0.set_ylim(-0.4, 10)
a0.axis("off")
a0.set_title("a  Decision tree (5 leaves)", x=0.02)
ys = [8.7, 6.6, 4.5, 2.4]
for i, (feat, thr, cls, nn) in enumerate(chain):
    a0.add_patch(
        FancyBboxPatch((0.4, ys[i] - 0.55), 3.9, 1.1, boxstyle="round,pad=0.05,rounding_size=0.15", fc="white", ec=INK, lw=1.2)
    )
    a0.text(2.35, ys[i], f"{feat} > {thr:.0f}% ?", ha="center", va="center", fontsize=11, fontweight="bold")
    a0.annotate("", xy=(5.9, ys[i]), xytext=(4.4, ys[i]), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    a0.text(5.15, ys[i] + 0.22, "yes", ha="center", fontsize=9, color=MUTE)
    a0.add_patch(
        FancyBboxPatch((6.0, ys[i] - 0.5), 3.7, 1.0, boxstyle="round,pad=0.05,rounding_size=0.15", fc=COL[cls], ec="none")
    )
    a0.text(
        7.85,
        ys[i],
        cls,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="white" if cls != "Gas-dominated" else INK,
    )
    yn = ys[i + 1] + 0.6 if i < 3 else 0.95
    a0.annotate("", xy=(2.35, yn), xytext=(2.35, ys[i] - 0.6), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    a0.text(2.6, (ys[i] - 0.6 + yn) / 2, "no", ha="left", va="center", fontsize=9, color=MUTE)
a0.add_patch(FancyBboxPatch((0.5, -0.05), 3.7, 1.0, boxstyle="round,pad=0.05,rounding_size=0.15", fc=COL[final[0]], ec="none"))
a0.text(2.35, 0.45, final[0], ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
a1 = fig.add_axes([0.485, 0.22, 0.175, 0.50])
P = PI.sort_values("imp_mean")
yy = np.arange(len(P))
top4 = P.imp_mean.values > 0.1
a1.hlines(yy, P.p05, P.p95, color=[ACC if s else CTX for s in top4], lw=2)
a1.scatter(P.imp_mean, yy, s=45, color=[ACC if s else MUTE for s in top4], zorder=3)
a1.set_yticks(yy)
a1.set_yticklabels(P.feature, fontsize=10, color=INK)
a1.axvline(0, color="#888888", lw=0.8)
a1.set_xlim(-0.05, 0.45)
a1.set_xlabel("Drop in hold-out balanced\naccuracy when shuffled", fontsize=9.5)
a1.tick_params(axis="y", length=0)
a1.set_title("b  Random forest:\npermutation importance", fontsize=11)
a1.xaxis.grid(True, color=LIGHT, lw=0.8)
a1.set_axisbelow(True)
a2 = fig.add_axes([0.755, 0.22, 0.225, 0.50])
cols = PI.feature.tolist()
M = CO[cols].T.values
im = a2.imshow(M, cmap="PuOr_r", vmin=-2.2, vmax=2.2, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        a2.text(
            j, i, fmt_signed(M[i, j], 1), ha="center", va="center", fontsize=8.5, color="white" if abs(M[i, j]) > 1.3 else INK
        )
a2.set_yticks(range(len(cols)))
a2.set_yticklabels(cols, fontsize=10, color=INK)
a2.set_xticks(range(len(ORDER)))
a2.set_xticklabels(
    [{"Diversified": "Divers."}.get(g_.split("-")[0], g_.split("-")[0]) for g_ in ORDER],
    fontsize=10,
    rotation=0,
    ha="center",
    color=INK,
)
a2.tick_params(length=0, pad=7)
for j_, g_ in enumerate(ORDER):
    a2.add_patch(plt.Rectangle((j_ - 0.5 + 0.04, len(cols) - 0.5 + 0.06), 0.92, 0.14, color=COL[g_], clip_on=False))
for s in a2.spines.values():
    s.set_visible(False)
a2.set_xticks(np.arange(-0.5, len(ORDER), 1), minor=True)
a2.set_yticks(np.arange(-0.5, len(cols), 1), minor=True)
a2.grid(which="minor", color="white", lw=1.5)
a2.tick_params(which="minor", length=0)
a2.set_title("c  Penalised logistic regression:\nlog-odds per +1 SD of share", fontsize=11)
save(fig, "F12_surrogates")

# ---------------- F13 transition matrix ----------------
TM = pd.read_csv(f"{O}/transition_matrix.csv", index_col=0).loc[ORDER, ORDER]
r4 = R["rq4"]
fig, ax = plt.subplots(figsize=(11.5, 6.3))
fig.subplots_adjust(left=0.22, right=0.70, top=0.68, bottom=0.10)
ax.imshow(TM.values, cmap="Greys", vmin=0, vmax=90, aspect="auto")
for i in range(5):
    for j in range(5):
        v = TM.values[i, j]
        if i == j:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none", ec=COL[ORDER[i]], lw=3, zorder=3))
        ax.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold" if i == j else "normal",
            color="white" if v > 50 else (INK if v > 0 else "#b0b0b0"),
        )
ax.set_xticks(range(5))
ax.set_xticklabels([g_.replace("-", "-\n") for g_ in ORDER], fontsize=10, color=INK)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel("Archetype in 2023", fontsize=11, labelpad=10)
ax.set_yticks(range(5))
ax.set_yticklabels([f"{g_}  ({TM.loc[g_].sum()})" for g_ in ORDER], fontsize=10.5, color=INK)
ax.set_ylabel("Archetype in 2000 (systems)", fontsize=11)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="minor", length=0)
notes = [
    ("Coal → Diversified: 10", "United States, Germany, Spain,\nSouth Korea, Greece, Portugal …"),
    ("Oil → Gas: 13", "Mexico, Iraq, Kuwait, Singapore,\nLibya, Azerbaijan …"),
    ("Hydro → elsewhere: 16", "hydro share fell as demand grew:\nVietnam → coal, Chile → diversified"),
    ("Diversified → Coal: 1", "Japan: nuclear share fell\nfrom 29% to 8%"),
]
for i, (h_, b_) in enumerate(notes):
    fig.text(0.725, 0.60 - i * 0.135, h_, fontsize=10.5, fontweight="bold", va="top")
    fig.text(0.725, 0.60 - i * 0.135 - 0.038, b_, fontsize=9.5, color=MUTE, va="top")
head(
    fig,
    f"Mixes are sticky: {r4['stayed']} of {r4['n_both']} power systems ({r4['stay_pct']:.0f}%) are in the same archetype as in 2000",
    "Rows = archetype in 2000, columns = archetype in 2023, cell = number of power systems. Outlined diagonal = no change. Grey intensity repeats the count.",
    src="2000 archetypes are derived: each system's 2000 mix is assigned to the nearest 2023 cluster centre, so the categories mean the same in both years. "
    + SRC,
)
save(fig, "F13_transitions")

# ---------------- F14 dumbbell ----------------
top = pd.read_csv(f"{O}/top20_dumbbell.csv").sort_values("ci_change", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11.5, 7.4))
fig.subplots_adjust(left=0.16, right=0.76, top=0.80, bottom=0.12)
for i, r in top.iterrows():
    c = COL[r.archetype]
    ax.plot([r.carbon_intensity_elec_2000, r.carbon_intensity_elec], [i, i], color=CTX, lw=2, zorder=1)
    ax.scatter(r.carbon_intensity_elec_2000, i, s=60, facecolor="white", edgecolor=MUTE, lw=1.5, zorder=3)
    ax.scatter(r.carbon_intensity_elec, i, s=75, color=c, zorder=4)
    moved = "" if r.archetype == r.archetype_2000 else f"  (was {r.archetype_2000.split('-')[0].lower()})"
    ax.text(
        1.02, i, f"{fmt_signed(r.ci_change)}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9.5, color=INK
    )
    ax.text(
        1.085,
        i,
        f"{r.archetype.split('-')[0]}{moved}",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=9.5,
        color=INK,
    )
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.country, fontsize=10.5, color=INK)
ax.set_ylim(len(top) - 0.4, -0.6)
ax.set_xlim(0, 850)
ax.set_xlabel("Carbon intensity † (gCO₂e per kWh)")
ax.xaxis.grid(True, color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.text(
    1.02,
    -1.0,
    "Change    2023 archetype",
    transform=ax.get_yaxis_transform(),
    fontsize=9.5,
    fontweight="bold",
    va="center",
    ha="left",
)
hs = [
    Line2D([], [], marker="o", ls="", mfc="white", mec=MUTE, mew=1.5, ms=8, label="2000"),
    Line2D([], [], marker="o", ls="", color=INK, ms=8, label="2023 (colour = archetype)"),
]
ax.legend(handles=hs, loc="lower right", fontsize=9.5, bbox_to_anchor=(1.0, 0.0), handletextpad=0.3)
ndown = (top.ci_change < 0).sum()
head(
    fig,
    f"{ndown} of the 20 largest power systems got cleaner; Vietnam and Japan got markedly dirtier",
    f"Carbon intensity of electricity in 2000 (hollow) and 2023 (filled), 20 largest generators = {r4['top20_gen_share']:.0f}% of world electricity. Sorted from largest rise to largest fall.",
    src=SRC + "  † Estimated from the generation mix with IPCC emission factors.",
)
save(fig, "F14_dumbbell")

# ---------------- F15 EU small multiples ----------------
eu6 = R["eu"]["top6"]
pe = panel[panel.country.isin(eu6) & (panel.year <= 2024)]
fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, sharey=True)
fig.subplots_adjust(left=0.07, right=0.975, top=0.80, bottom=0.11, wspace=0.10, hspace=0.26)
arch = x.set_index("country").archetype
for a, cn in zip(axs.ravel(), eu6):
    for other in eu6:
        if other != cn:
            o = pe[pe.country == other]
            a.plot(o.year, o.carbon_intensity_elec, color=LIGHT, lw=1.2, zorder=1)
    s = pe[pe.country == cn].sort_values("year")
    c = COL[arch[cn]]
    a.plot(s.year, s.carbon_intensity_elec, color=c, lw=2.6, zorder=3)
    v0, v1 = s.carbon_intensity_elec.iloc[0], s.carbon_intensity_elec.iloc[-1]
    a.scatter([s.year.iloc[0], s.year.iloc[-1]], [v0, v1], color=c, s=28, zorder=4)
    a.set_title(f"{cn}", fontsize=12.5, color=INK, loc="left")
    a.text(
        1.0,
        1.035,
        f"{v0:.0f} → {v1:.0f}  ({fmt_signed(100*(v1/v0-1))}%)",
        transform=a.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=INK,
        fontweight="bold",
    )
    a.text(0.985, 0.95, arch[cn], transform=a.transAxes, ha="right", va="top", fontsize=9.5, color=MUTE)
    a.set_ylim(0, 1000)
    a.set_xlim(2000, 2024)
    a.set_xticks([2000, 2010, 2020])
    a.yaxis.grid(True, color=LIGHT, lw=0.8)
    a.set_axisbelow(True)
for a in axs[:, 0]:
    a.set_ylabel("gCO₂e per kWh †")
head(
    fig,
    "Italy cut the carbon intensity of its power by 44% without leaving the gas archetype",
    "Carbon intensity of electricity, 2000–2024, six largest EU power systems. Same axes in every panel; grey lines = the other five. Line colour and label = 2023 archetype; header = 2000 value → 2024 value (change).",
    src=SRC + "  † Estimated from the generation mix with IPCC emission factors.",
)
save(fig, "F15_eu")

if SLIDES:
    json.dump(TITLES, open(f"{F}/titles.json", "w"), indent=1, ensure_ascii=False)
