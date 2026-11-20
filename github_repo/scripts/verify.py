"""Step 5 - independent re-computation of the headline numbers.

Everything below is recomputed straight from the RAW file (plus the archetype labels) with separate code,
then compared with what the pipeline stored in outputs/results.json and the CSV tables.
Exit code 0 = every check passed.

Usage:  python scripts/verify.py [DATA_DIR] [OUTPUT_DIR]     (defaults: data/  outputs/)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data")
O = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "outputs")

R = json.load(open(f"{O}/results.json"))
raw = pd.read_csv(f"{D}/owid_energy/owid-energy-data.csv")
raw.loc[raw.country == "Kosovo", "iso_code"] = "OWID_KOS"  # same recode as prep.py
labels = pd.read_csv(f"{O}/xs2023_labeled.csv")[["iso_code", "archetype"]]
income = pd.read_csv(f"{D}/other/income_groups.csv").rename(columns={"Code": "iso_code", "Income group": "income"})
INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]

checks = []


def check(name, got, expected, tol=1e-6):
    ok = bool(np.isclose(got, expected, rtol=tol, atol=tol))
    checks.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: recomputed {got:.6g} | pipeline {expected:.6g}")


def eta2_h(h, k, n):
    """Kruskal-Wallis effect size eta-squared (Tomczak & Tomczak 2014)."""
    return (h - k + 1) / (n - k)


# ---- RQ1: world totals --------------------------------------------------------------------------
w = raw[raw.country == "World"].set_index("year")
check(
    "world generation change 2000-2024 (%)",
    100 * (w.loc[2024, "electricity_generation"] / w.loc[2000, "electricity_generation"] - 1),
    R["rq1"]["gen_chg_pct"],
)
check(
    "world carbon-intensity change (%)",
    100 * (w.loc[2024, "carbon_intensity_elec"] / w.loc[2000, "carbon_intensity_elec"] - 1),
    R["rq1"]["ci_chg_pct"],
)
check(
    "world power-emissions change (%)",
    100 * (w.loc[2024, "greenhouse_gas_emissions"] / w.loc[2000, "greenhouse_gas_emissions"] - 1),
    R["rq1"]["em_chg_pct"],
)

# ---- sample sizes -------------------------------------------------------------------------------
snap = raw[(raw.year == 2023) & raw.iso_code.notna() & (raw.electricity_generation > 0)].merge(labels, on="iso_code")
check("2023 snapshot n", len(snap), R["sample_sizes"]["2023 snapshot"])
snap_inc = snap.merge(income[["iso_code", "income"]], on="iso_code")
check("n with income group", len(snap_inc), R["sample_sizes"]["With World Bank income group"])
gdp22 = raw[(raw.year == 2022) & raw.gdp.notna()][["iso_code", "gdp", "population"]].assign(gdp_pc=lambda t: t.gdp / t.population)
snap_gdp = snap.merge(gdp22[["iso_code", "gdp_pc"]], on="iso_code")
check("n with GDP per capita", len(snap_gdp), R["sample_sizes"]["With GDP per capita (2022)"])

# ---- archetype sizes and generation shares ------------------------------------------------------
for a, n in R["cluster"]["sizes"].items():
    check(f"archetype size: {a}", (snap.archetype == a).sum(), n)
summary = pd.read_csv(f"{O}/cluster_summary.csv", index_col=0)
share = 100 * snap.groupby("archetype").electricity_generation.sum() / snap.electricity_generation.sum()
for a in share.index:
    check(f"share of world generation: {a} (%)", share[a], summary.loc[a, "gen_share"])

# ---- external tests: Kruskal-Wallis + Benjamini-Hochberg over the 8 external variables ----------
ext = pd.read_csv(f"{O}/tests_external.csv").set_index("col")
order = R["order"]
numeric = {
    "carbon_intensity_elec": snap,
    "electricity_generation": snap,
    "population": snap,
    "per_capita_electricity": snap,
    "net_elec_imports_share_demand": snap,
}
p_re = {}
for col, frame in numeric.items():
    d = frame[["archetype", col]].dropna()
    h, p = stats.kruskal(*[d[d.archetype == a][col] for a in order])
    p_re[col] = p
    check(f"KW archetype ~ {col}: H", h, ext.loc[col, "stat"])
    check(f"KW archetype ~ {col}: eta2", eta2_h(h, 5, len(d)), ext.loc[col, "effect"])
h, p = stats.kruskal(*[snap_gdp[snap_gdp.archetype == a].gdp_pc for a in order])
p_re["gdp_pc_latest"] = p
check("KW archetype ~ GDP per capita: H", h, ext.loc["gdp_pc_latest", "stat"])
# the two chi-square p-values are Monte-Carlo estimates: reuse them, but recompute the BH adjustment
p_all = [p_re.get(c, ext.loc[c, "p"]) for c in ext.index]
q_re = multipletests(p_all, method="fdr_bh")[1]
check("number of tests in the BH family", len(p_all), 8)
for c, q in zip(ext.index, q_re):
    check(f"BH q (m = 8): {c}", q, ext.loc[c, "q_BH"], tol=1e-4)

# ---- RQ3: income --------------------------------------------------------------------------------
h, p = stats.kruskal(*[snap_inc[snap_inc.income == g].carbon_intensity_elec for g in INCOME_ORDER])
check("RQ3 KW income ~ carbon intensity: H", h, R["rq3"]["kw_H"])
check("RQ3 KW income ~ carbon intensity: p", p, R["rq3"]["kw_p"])
check("RQ3 eta2", eta2_h(h, 4, len(snap_inc)), R["rq3"]["eta2_H"])
h, p = stats.kruskal(*[snap_inc[snap_inc.income == g].per_capita_electricity for g in INCOME_ORDER])
check("RQ3 KW income ~ kWh per person: H", h, R["rq3"]["kw_kwh"]["H"])
check(
    "RQ3 Spearman GDP ~ carbon intensity",
    stats.spearmanr(snap_gdp.gdp_pc, snap_gdp.carbon_intensity_elec)[0],
    R["rq3"]["spearman_gdp_ci"]["rho"],
)
check(
    "RQ3 Spearman GDP ~ kWh per person",
    stats.spearmanr(snap_gdp.gdp_pc, snap_gdp.per_capita_electricity)[0],
    R["rq3"]["spearman_gdp_kwh"]["rho"],
)

# ---- RQ4: transitions and Italy -----------------------------------------------------------------
tm = pd.read_csv(f"{O}/transition_matrix.csv", index_col=0)
check("systems in both 2000 and 2023", tm.values.sum(), R["sample_sizes"]["In both 2000 and 2023"])
check("systems in the same archetype", np.trace(tm.loc[order, order].values), R["rq4"]["stayed"])
it = raw[raw.country == "Italy"].set_index("year")
check("Italy carbon intensity 2000", it.loc[2000, "carbon_intensity_elec"], R["italy"]["ci_2000"])
check("Italy carbon intensity 2024", it.loc[2024, "carbon_intensity_elec"], R["italy"]["ci_2024"])

print(f"\n{sum(checks)} of {len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
