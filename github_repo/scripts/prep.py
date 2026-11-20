"""Step 1 - data audit ("smell test") and cleaning.

Reads the raw Our World in Data energy file plus the enrichment tables, removes aggregates,
restricts the time window, repairs structural zeros, joins income group / continent / UN region,
and writes the clean panel, the 2023 snapshot and the cleaning log.

Usage:  python scripts/prep.py [DATA_DIR] [OUTPUT_DIR]      (defaults: data/  outputs/)
Outputs: panel_clean.csv, xs2023.csv, world.csv, coverage_by_year.csv, cleaning_log.csv, structural_zero.json
"""

import sys, json, pandas as pd, numpy as np

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
D = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data")
O = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "outputs")
Path(O).mkdir(parents=True, exist_ok=True)
log = []  # (step, action, reason, rows_before, rows_after)
e = pd.read_csv(f"{D}/owid_energy/owid-energy-data.csv")
raw_shape = e.shape
SH = [
    "coal_share_elec",
    "gas_share_elec",
    "oil_share_elec",
    "nuclear_share_elec",
    "hydro_share_elec",
    "wind_share_elec",
    "solar_share_elec",
    "biofuel_share_elec",
    "other_renewables_share_elec_exc_biofuel",
]
NAMES = ["Coal", "Gas", "Oil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other renewables"]
KEEP = [
    "country",
    "iso_code",
    "year",
    "population",
    "gdp",
    "electricity_generation",
    "carbon_intensity_elec",
    "greenhouse_gas_emissions",
    "per_capita_electricity",
    "net_elec_imports_share_demand",
    "low_carbon_share_elec",
    "fossil_share_elec",
    "renewables_share_elec",
] + SH

# 1 Kosovo: real country without ISO code in the file
print("Kosovo rows with generation:", e[(e.country == "Kosovo") & e.electricity_generation.notna()].shape[0])
e.loc[e.country == "Kosovo", "iso_code"] = "OWID_KOS"
# 2 aggregates / historical
agg = e[e.iso_code.isna()]
world = e[e.country == "World"].copy()
n0 = len(e)
c = e[e.iso_code.notna()].copy()
log.append(
    (
        "Remove aggregate & historical entities",
        f"dropped {agg.country.nunique()} entities without ISO code (regions, income groups, OECD/G20, USSR…); 'World' row kept separately for the global trend",
        "aggregates mixed with country rows would double-count",
        n0,
        len(c),
    )
)
# 3 columns
n_cols = c.shape[1]
c = c[KEEP]
world = world[KEEP]
log.append(
    (
        "Column selection",
        f"kept {len(KEEP)} of {n_cols} columns",
        "other 100+ columns are primary-energy / per-fuel consumption metrics outside the question (and affected by OWID's primary-energy method change)",
        len(c),
        len(c),
    )
)
# 4 time window
cov = e[e.iso_code.notna()].groupby("year").apply(lambda d: (d.electricity_generation.notna()).sum())
cov.to_csv(f"{O}/coverage_by_year.csv")
n1 = len(c)
c = c[(c.year >= 2000) & (c.year <= 2024)]
world = world[(world.year >= 2000) & (world.year <= 2024)]
log.append(
    (
        "Time window 2000–2024",
        "dropped years <2000 and 2025",
        f"coverage jumps from {cov[1999]} to {cov[2000]} countries in 2000 (Ember series start); 2025 has only {cov[2025]} countries (partial year)",
        n1,
        len(c),
    )
)
# 5 rows w/o generation
n2 = len(c)
c = c[c.electricity_generation.notna()]
log.append(("No electricity data", "dropped country-years with missing generation", "nothing to analyse", n2, len(c)))
# 6 zero generation
n3 = len(c)
z = c[c.electricity_generation <= 0]
print("zero-generation rows:", z[["country", "year"]].values.tolist()[:20], len(z))
c = c[c.electricity_generation > 0]
log.append(
    (
        "Zero generation",
        "dropped country-years with generation = 0 TWh (rounded)",
        f"shares undefined (0/0); affects {z.country.nunique()} tiny systems, e.g. Niue, Lesotho 2023–24",
        n3,
        len(c),
    )
)
# 7 structural zeros
nmiss = c[SH].isna().sum(axis=1)
resid = 100 - c[SH].sum(axis=1)
fixable = (nmiss > 0) & (resid.abs() < 0.05)
unfix = (nmiss > 0) & ~fixable
print(
    "rows with ≥1 missing share:", (nmiss > 0).sum(), "| closure holds (fixable):", fixable.sum(), "| not fixable:", unfix.sum()
)
print(c[unfix][["country", "year"] + SH].head(20))
y23 = c[c.year == 2023]
lost = y23[y23[SH].isna().any(axis=1)]
print(
    "2023: naive dropna would drop %d systems = %.1f%% of generation; e.g. %s"
    % (
        len(lost),
        100 * lost.electricity_generation.sum() / y23.electricity_generation.sum(),
        lost.sort_values("electricity_generation", ascending=False).country.head(8).tolist(),
    )
)
json.dump(
    {
        "naive_drop_n": int(len(lost)),
        "naive_drop_gen_share": float(lost.electricity_generation.sum() / y23.electricity_generation.sum()),
        "cells_filled": int(c.loc[fixable, SH].isna().sum().sum()),
        "rows_filled": int(fixable.sum()),
    },
    open(f"{O}/structural_zero.json", "w"),
)
c["imputed_zero"] = fixable
c.loc[fixable, SH] = c.loc[fixable, SH].fillna(0)
n4 = len(c)
c = c[~unfix]
log.append(
    (
        "Structural zeros",
        f"NaN → 0 in {int(fixable.sum())} country-years where the observed shares already sum to 100 % (|residual| < 0.05 pp)",
        "source stores 'no such plant' as missing; plain dropna() would delete China, India, Japan, Russia, Brazil…",
        n4,
        len(c),
    )
)
# closure check
s = c[SH].sum(axis=1)
print("closure after fix: min %.3f max %.3f" % (s.min(), s.max()))
# 8 joins
inc = pd.read_csv(f"{D}/other/income_groups.csv").rename(columns={"Code": "iso_code", "Income group": "income_group"})[
    ["iso_code", "income_group"]
]
print("income dup codes:", inc.iso_code.duplicated().sum(), inc.income_group.value_counts().to_dict())
cont = pd.read_csv(f"{D}/other/continents.csv")
print(cont.columns.tolist(), cont.head(2).values.tolist())
cc = pd.read_csv(f"{D}/other/country-codes.csv")[
    ["ISO3166-1-Alpha-3", "Region Name", "Sub-region Name", "Intermediate Region Name"]
].rename(
    columns={
        "ISO3166-1-Alpha-3": "iso_code",
        "Region Name": "un_region",
        "Sub-region Name": "un_subregion",
        "Intermediate Region Name": "un_intermediate",
    }
)
ccol = [k for k in cont.columns if "ode" in k][0]
rcol = (
    [k for k in cont.columns if "ontinent" in k.lower() and "ode" not in k][0]
    if any("ontinent" in k.lower() for k in cont.columns)
    else cont.columns[-1]
)
cont = cont.rename(columns={ccol: "iso_code", rcol: "continent"})[["iso_code", "continent"]].drop_duplicates("iso_code")
c = c.merge(inc, on="iso_code", how="left").merge(cont, on="iso_code", how="left").merge(cc, on="iso_code", how="left")
y = c[c.year == 2023]
print(
    "2023 systems:",
    len(y),
    "| no income group:",
    y.income_group.isna().sum(),
    sorted(y[y.income_group.isna()].country),
    "| no continent:",
    sorted(y[y.continent.isna()].country),
)
log.append(
    (
        "External join 1",
        "left-join World Bank income group (2020 vintage) on ISO-3",
        f"enrichment for RQ3; {int(y.income_group.isna().sum())} territories without a WB class keep NaN and are excluded only from income tests",
        len(c),
        len(c),
    )
)
log.append(
    (
        "External join 2",
        "left-join OWID continent + UN M49 sub-region on ISO-3",
        "regional context for labels and the Europe focus",
        len(c),
        len(c),
    )
)
# 9 derived
c["gdp_pc"] = c.gdp / c.population
gl = (
    c[c.gdp_pc.notna()]
    .sort_values("year")
    .groupby("iso_code")
    .tail(1)[["iso_code", "year", "gdp_pc"]]
    .rename(columns={"year": "gdp_year", "gdp_pc": "gdp_pc_latest"})
)
print("gdp latest year distribution:", gl.gdp_year.value_counts().to_dict())
c = c.merge(gl, on="iso_code", how="left")
c["power_emis_calc"] = c.carbon_intensity_elec * c.electricity_generation / 1000  # MtCO2e
log.append(
    (
        "Derived variables",
        "GDP per capita = gdp / population (latest available year ≤ 2022, Maddison 2011 int-$)",
        "external explanatory variable; GDP is not published for 2023+, so it is carried forward and flagged",
        len(c),
        len(c),
    )
)
c.rename(columns=dict(zip(SH, NAMES)), inplace=True)
world.rename(columns=dict(zip(SH, NAMES)), inplace=True)
c.to_csv(f"{O}/panel_clean.csv", index=False)
world.to_csv(f"{O}/world.csv", index=False)
x = c[c.year == 2023].copy()
x.to_csv(f"{O}/xs2023.csv", index=False)
print(
    "panel",
    c.shape,
    "| systems",
    c.iso_code.nunique(),
    "| xs2023",
    x.shape,
    "| gdp_pc available in xs:",
    x.gdp_pc_latest.notna().sum(),
    "| CI missing:",
    x.carbon_intensity_elec.isna().sum(),
)
print("per-year n complete:", c.groupby("year").size().to_dict())
L = pd.DataFrame(log, columns=["step", "action", "reason", "rows_before", "rows_after"])
L.to_csv(f"{O}/cleaning_log.csv", index=False)
print(L.to_string())
print("raw shape", raw_shape)
# smell-test summary
print(
    "xs describe:\n",
    x[
        NAMES
        + [
            "carbon_intensity_elec",
            "electricity_generation",
            "per_capita_electricity",
            "gdp_pc_latest",
            "net_elec_imports_share_demand",
        ]
    ]
    .describe()
    .T.round(2),
)
print(x.sort_values("carbon_intensity_elec").tail(5)[["country", "carbon_intensity_elec", "Coal", "Oil", "Gas"]])
print(x.sort_values("per_capita_electricity").tail(4)[["country", "per_capita_electricity"]])
print(
    x.sort_values("net_elec_imports_share_demand").iloc[[0, 1, 2, -3, -2, -1]][
        ["country", "net_elec_imports_share_demand", "electricity_generation"]
    ]
)
