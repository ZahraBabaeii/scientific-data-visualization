"""Step 2 - statistics, dimensionality reduction, clustering and cluster explanation.

Pipeline (course lectures 5-7):
  PCA -> choice of k (silhouette + bootstrap ARI) -> k-means (k = 5) -> k +/- 1 and second-method checks
  -> UMAP (PCA initialisation, seeds / hyper-parameters varied) -> 6-step explanation:
     1 summarise, 2 test, 3 adjust (Benjamini-Hochberg over the 8 EXTERNAL variables only),
     4 localise (Dunn + Holm), 5 surrogate models (tree, random forest, penalised logistic), 6 show (figs.py)
  -> RQ3 income tests -> RQ4 transitions 2000 -> 2023.

All seeds are fixed: k-means random_state=0, bootstrap seed 42, UMAP seed 0, train/test split seed 0.

Usage:  python scripts/analysis.py [OUTPUT_DIR]             (default: outputs/)
Outputs: results.json plus one CSV per table (see README).
"""

import sys, json, warnings, itertools, pandas as pd, numpy as np

warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import umap

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
O = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs")
R = {}
N = ["Coal", "Gas", "Oil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other renewables"]
x = pd.read_csv(f"{O}/xs2023.csv")
panel = pd.read_csv(f"{O}/panel_clean.csv")
world = pd.read_csv(f"{O}/world.csv").set_index("year")
X = x[N].values
n = len(x)
K = 5

# ---------- RQ1 world ----------
w = world
a, b = w.loc[2000], w.loc[2024]
R["rq1"] = {
    "gen_2000": a.electricity_generation,
    "gen_2024": b.electricity_generation,
    "gen_chg_pct": 100 * (b.electricity_generation / a.electricity_generation - 1),
    "ci_2000": a.carbon_intensity_elec,
    "ci_2024": b.carbon_intensity_elec,
    "ci_chg_pct": 100 * (b.carbon_intensity_elec / a.carbon_intensity_elec - 1),
    "em_2000": a.greenhouse_gas_emissions,
    "em_2024": b.greenhouse_gas_emissions,
    "em_chg_pct": 100 * (b.greenhouse_gas_emissions / a.greenhouse_gas_emissions - 1),
    "ci_peak_year": int(w.carbon_intensity_elec.idxmax()),
    "ci_peak": w.carbon_intensity_elec.max(),
    "shares_2000": a[N].round(2).to_dict(),
    "shares_2024": b[N].round(2).to_dict(),
    "lowcarbon_2000": a.low_carbon_share_elec,
    "lowcarbon_2024": b.low_carbon_share_elec,
    "em_peak_year": int(w.greenhouse_gas_emissions.idxmax()),
}
print("RQ1", json.dumps(R["rq1"], default=float, indent=0)[:900])

# ---------- PCA ----------
pca = PCA().fit(X)
Z = pca.transform(X)
R["pca"] = {
    "var_ratio": pca.explained_variance_ratio_.round(4).tolist(),
    "std_var_ratio": PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_.round(4).tolist(),
}
load = pd.DataFrame(pca.components_[:3].T, index=N, columns=["PC1", "PC2", "PC3"])
load.to_csv(f"{O}/pca_loadings.csv")
x["PC1"], x["PC2"], x["PC3"] = Z[:, 0], Z[:, 1], Z[:, 2]

# ---------- choose k ----------
rng = np.random.default_rng(42)
rows = []
for k in range(2, 11):
    km = KMeans(k, n_init=50, random_state=0).fit(X)
    lab = km.labels_
    aris = []
    for bb in range(200):
        idx = rng.choice(n, n, replace=True)
        aris.append(adjusted_rand_score(lab, KMeans(k, n_init=10, random_state=bb).fit(X[idx]).predict(X)))
    rows.append(
        dict(
            k=k,
            inertia=km.inertia_,
            silhouette=silhouette_score(X, lab),
            ari_mean=np.mean(aris),
            ari_p05=np.percentile(aris, 5),
            ari_p95=np.percentile(aris, 95),
            min_size=int(np.bincount(lab).min()),
        )
    )
ksel = pd.DataFrame(rows)
ksel.to_csv(f"{O}/k_selection.csv", index=False)
print(ksel.round(3).to_string())

km = KMeans(K, n_init=100, random_state=0).fit(X)
raw = km.labels_
cen = pd.DataFrame(km.cluster_centers_, columns=N)


def name(c):
    top = c.idxmax()
    return (
        {"Oil": "Oil-dependent", "Hydro": "Hydro-dominated", "Coal": "Coal-heavy", "Gas": "Gas-dominated"}[top]
        if c.max() > 40
        else "Diversified"
    )


names = [name(cen.loc[i]) for i in range(K)]
assert len(set(names)) == K, names
x["archetype"] = [names[i] for i in raw]
ORDER = x.groupby("archetype").carbon_intensity_elec.median().sort_values().index.tolist()
R["order"] = ORDER
print("ORDER (by median CI):", ORDER)
x["sil"] = silhouette_samples(X, raw)
R["cluster"] = {
    "k": K,
    "silhouette": silhouette_score(X, raw),
    "sizes": x.archetype.value_counts().to_dict(),
    "ari_ward": adjusted_rand_score(raw, AgglomerativeClustering(K, linkage="ward").fit_predict(X)),
    "ari_hellinger": adjusted_rand_score(raw, KMeans(K, n_init=50, random_state=0).fit(np.sqrt(X / 100)).labels_),
    "ari_zscore": adjusted_rand_score(raw, KMeans(K, n_init=50, random_state=0).fit(StandardScaler().fit_transform(X)).labels_),
    "ari_k4": adjusted_rand_score(raw, KMeans(4, n_init=50, random_state=0).fit(X).labels_),
    "ari_k6": adjusted_rand_score(raw, KMeans(6, n_init=50, random_state=0).fit(X).labels_),
    "neg_sil_n": int((x.sil < 0).sum()),
    "sil_by_cluster": x.groupby("archetype").sil.mean().round(3).to_dict(),
}
R["cluster"].update(
    {f"k{r.k}": {"sil": r.silhouette, "ari": r.ari_mean, "ari_p05": r.ari_p05} for r in ksel.itertuples() if r.k in (4, 5, 6)}
)
print(json.dumps(R["cluster"], default=float))
# ---------- k +/- 1 sensitivity and agreement with other clustering methods ----------
# k = 4 and k = 6 are cross-tabulated against the chosen k = 5 archetypes so we can say which groups merge or split.
ksens = {}
for kk in (4, 6):
    alt = KMeans(kk, n_init=50, random_state=0).fit(X)
    xt = pd.crosstab(pd.Series(x.archetype.values, name="k5_archetype"), pd.Series(alt.labels_, name=f"k{kk}_cluster"))
    cen_alt = pd.DataFrame(alt.cluster_centers_, columns=N)
    lab_alt = {
        j: "+".join(f"{c} {cen_alt.loc[j,c]:.0f}%" for c in cen_alt.loc[j].sort_values(ascending=False).index[:2])
        for j in range(kk)
    }
    xt.columns = [f"{j}: {lab_alt[j]} (n={int((alt.labels_==j).sum())})" for j in xt.columns]
    xt.to_csv(f"{O}/k_sensitivity_k{kk}.csv")
    print(xt)
    ksens[f"k{kk}"] = {
        "sizes": sorted(np.bincount(alt.labels_).tolist()),
        "ari_vs_k5": adjusted_rand_score(raw, alt.labels_),
        "silhouette": silhouette_score(X, alt.labels_),
        "top2_sources_per_cluster": lab_alt,
    }
ward = AgglomerativeClustering(K, linkage="ward").fit_predict(X)
xtw = pd.crosstab(pd.Series(x.archetype.values, name="kmeans_archetype"), pd.Series(ward, name="ward_cluster"))
xtw.to_csv(f"{O}/agreement_ward.csv")
print(xtw)
# Other hierarchical linkages, for completeness.
link_ari = {lk: adjusted_rand_score(raw, AgglomerativeClustering(K, linkage=lk).fit_predict(X)) for lk in ("average", "complete")}
# Gaussian mixtures on the first 4 principal components (95% of variance): the 9 shares are collinear
# (they sum to 100), so a covariance matrix on the raw shares is singular.
# "tied" = one covariance shared by all components (same assumption family as k-means);
# "full" = each component has its own covariance (components may differ wildly in spread).
gmm_ari = {}
for ct in ("tied", "full"):
    gm = GaussianMixture(K, covariance_type=ct, n_init=20, random_state=0, reg_covar=1e-3).fit(Z[:, :4])
    gl = gm.predict(Z[:, :4])
    gmm_ari[ct] = adjusted_rand_score(raw, gl)
    pd.crosstab(pd.Series(x.archetype.values, name="kmeans_archetype"), pd.Series(gl, name=f"gmm_{ct}_cluster")).to_csv(
        f"{O}/agreement_gmm_{ct}.csv"
    )
R["second_method"] = {
    "ward_ari": adjusted_rand_score(raw, ward),
    "ward_sizes": sorted(np.bincount(ward).tolist()),
    "average_linkage_ari": link_ari["average"],
    "complete_linkage_ari": link_ari["complete"],
    "gmm_tied_ari": gmm_ari["tied"],
    "gmm_full_ari": gmm_ari["full"],
    "gmm_spec": "GaussianMixture(k=5, n_init=20, reg_covar=1e-3, seed 0) fitted on PC1-PC4",
}
R["k_sensitivity"] = ksens
print("SECOND METHOD", json.dumps(R["second_method"], default=float))
print("K SENS", json.dumps(ksens, default=float))
centers = cen.copy()
centers.index = names
centers.loc[ORDER].to_csv(f"{O}/centroids.csv")

# ---------- UMAP ----------
init = Z[:, :2] / Z[:, 0].std() * 1e-0


def emb(seed, nn, md):
    return umap.UMAP(n_neighbors=nn, min_dist=md, init=init.copy(), random_state=seed, metric="euclidean").fit_transform(X)


def knn_sets(E, k=10):
    d = ((E[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return [set(r) for r in np.argsort(d, 1)[:, :k]]


U = emb(0, 15, 0.1)
x["U1"], x["U2"] = U[:, 0], U[:, 1]
ur = []
base = knn_sets(U)
for nn, md in [(10, 0.1), (15, 0.1), (30, 0.1), (15, 0.3)]:
    for seed in range(5):
        E = emb(seed, nn, md)
        ks = knn_sets(E)
        ur.append(
            dict(
                n_neighbors=nn,
                min_dist=md,
                seed=seed,
                trust=trustworthiness(X, E, n_neighbors=10),
                knn_jaccard_vs_ref=np.mean([len(p & q) / len(p | q) for p, q in zip(base, ks)]),
                sil_kmeans_labels=silhouette_score(E, raw),
            )
        )
ur = pd.DataFrame(ur)
ur.to_csv(f"{O}/umap_stability.csv", index=False)
print(ur.groupby(["n_neighbors", "min_dist"])[["trust", "knn_jaccard_vs_ref", "sil_kmeans_labels"]].agg(["mean", "min"]).round(3))
R["umap"] = {
    "ref": "n_neighbors=15, min_dist=0.1, metric=euclidean, init=PCA(2), seed=0",
    "trust_ref": float(trustworthiness(X, U, n_neighbors=10)),
    "trust_pca2": float(trustworthiness(X, Z[:, :2], n_neighbors=10)),
    "trust_min": float(ur.trust.min()),
    "trust_max": float(ur.trust.max()),
    "jacc_seed_mean": float(ur[(ur.n_neighbors == 15) & (ur.min_dist == 0.1) & (ur.seed > 0)].knn_jaccard_vs_ref.mean()),
    "jacc_all_min": float(ur[ur.seed > 0].knn_jaccard_vs_ref.min()),
}
print(R["umap"])

# ---------- Step 1 summarise ----------
tot_gen = x.electricity_generation.sum()
tot_pop = x.population.sum()
summ = (
    x.groupby("archetype")
    .agg(
        n=("country", "size"),
        gen_share=("electricity_generation", lambda s: 100 * s.sum() / tot_gen),
        pop_share=("population", lambda s: 100 * s.sum() / tot_pop),
        ci_median=("carbon_intensity_elec", "median"),
        ci_q1=("carbon_intensity_elec", lambda s: s.quantile(0.25)),
        ci_q3=("carbon_intensity_elec", lambda s: s.quantile(0.75)),
        kwh_pc_median=("per_capita_electricity", "median"),
        gdp_pc_median=("gdp_pc_latest", "median"),
        gdp_n=("gdp_pc_latest", "count"),
        median_gen_twh=("electricity_generation", "median"),
    )
    .loc[ORDER]
)
med = x.groupby("archetype")[N].median().loc[ORDER]
mean = x.groupby("archetype")[N].mean().loc[ORDER]
print(summ.round(1).to_string())
print("means\n", mean.round(1).to_string())
print("medians\n", med.round(1).to_string())
summ.to_csv(f"{O}/cluster_summary.csv")
mean.to_csv(f"{O}/cluster_means.csv")
med.to_csv(f"{O}/cluster_medians.csv")
zprof = (mean - x[N].mean()) / x[N].std()
zprof.to_csv(f"{O}/cluster_zprofile.csv")
for a_ in ORDER:
    d = x[x.archetype == a_].sort_values("electricity_generation", ascending=False)
    print(a_, "| largest:", d.country.head(12).tolist())
print("Italy:", x[x.country == "Italy"][["archetype", "carbon_intensity_elec"] + N].round(1).to_dict("records"))


# ---------- Step 2-3 tests + BH (external variables only) ----------
def eta2_H(H, k, n_):
    return (H - k + 1) / (n_ - k)


def cramers_v(tab):
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n_ = tab.values.sum()
    r, c = tab.shape
    return chi2, np.sqrt(chi2 / (n_ * (min(r, c) - 1)))


def perm_chi2(a_, b_, B=20000, seed=1):
    rg = np.random.default_rng(seed)
    tab = pd.crosstab(a_, b_)
    obs, v = cramers_v(tab)
    cnt = 0
    bv = b_.values.copy()
    for _ in range(B):
        rg.shuffle(bv)
        t = pd.crosstab(a_.values, bv)
        if cramers_v(t)[0] >= obs - 1e-12:
            cnt += 1
    return obs, v, (cnt + 1) / (B + 1), tab


def eta2_H(H, k, n_):
    return (H - k + 1) / (n_ - k)


def cramers_v(tab):
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n_ = tab.values.sum()
    r, c = tab.shape
    return chi2, np.sqrt(chi2 / (n_ * (min(r, c) - 1)))


def perm_chi2(a_, b_, B=20000, seed=1):
    rg = np.random.default_rng(seed)
    tab = pd.crosstab(a_, b_)
    obs, v = cramers_v(tab)
    cnt = 0
    bv = b_.values.copy()
    for _ in range(B):
        rg.shuffle(bv)
        t = pd.crosstab(a_.values, bv)
        if cramers_v(t)[0] >= obs - 1e-12:
            cnt += 1
    return obs, v, (cnt + 1) / (B + 1), tab


# The 9 shares BUILT the clusters, so testing them would be circular: for them we keep only a descriptive
# separation index (Kruskal-Wallis eta-squared, no p-values). Statistical tests and the Benjamini-Hochberg
# correction are run on the 8 EXTERNAL variables only. Carbon intensity is kept in that family but flagged:
# it is computed from the mix, so its association with the archetypes is expected rather than discovered.
desc = []
for v in N:
    gs = [x[x.archetype == a_][v].values for a_ in ORDER]
    H, _ = stats.kruskal(*gs)
    desc.append(
        dict(variable=v + " share", col=v, H=H, eta2_H=eta2_H(H, K, n), note="descriptive separation only (clustering input)")
    )
DESC = pd.DataFrame(desc).sort_values("eta2_H", ascending=False)
DESC.to_csv(f"{O}/descriptive_separation_inputs.csv", index=False)
print(DESC.round(3).to_string())
tests = []
cont_ext = {
    "carbon_intensity_elec": "Carbon intensity (gCO2e/kWh) †",
    "electricity_generation": "System size (TWh generated)",
    "population": "Population",
    "gdp_pc_latest": "GDP per capita (2022)",
    "per_capita_electricity": "Electricity per person (kWh)",
    "net_elec_imports_share_demand": "Net imports (% of demand)",
}
for v in cont_ext:
    d = x[["archetype", v]].dropna()
    gs = [d[d.archetype == a_][v].values for a_ in ORDER]
    H, p = stats.kruskal(*gs)
    tests.append(
        dict(
            variable=cont_ext[v],
            col=v,
            role="derived from the mix" if v == "carbon_intensity_elec" else "independent",
            test="Kruskal-Wallis",
            stat=H,
            df=K - 1,
            n=len(d),
            p=p,
            effect=eta2_H(H, K, len(d)),
            effect_name="eta2_H",
        )
    )
for v, lab in [("income_group", "World Bank income group"), ("continent", "Continent")]:
    d = x[["archetype", v]].dropna()
    chi2, V, p, tab = perm_chi2(d.archetype, d[v])
    tests.append(
        dict(
            variable=lab,
            col=v,
            role="independent",
            test="Chi-square (Monte-Carlo, 20,000 perms)",
            stat=chi2,
            df=(tab.shape[0] - 1) * (tab.shape[1] - 1),
            n=len(d),
            p=p,
            effect=V,
            effect_name="Cramér's V",
        )
    )
    print(tab.loc[ORDER])
    print("min expected:", stats.chi2_contingency(tab)[3].min().round(2))
    tab.loc[ORDER].to_csv(f"{O}/xtab_{v}.csv")
T = pd.DataFrame(tests)
T["q_BH"] = multipletests(T.p, method="fdr_bh")[1]
T["m_tests"] = len(T)
assert len(T) == 8
T.to_csv(f"{O}/tests_external.csv", index=False)
print(T.drop(columns=["col", "m_tests"]).sort_values("effect", ascending=False).to_string(float_format=lambda v: f"{v:.3g}"))
R["external_tests"] = {
    "m": int(len(T)),
    "method": "Benjamini-Hochberg across the 8 external variables",
    "q": {r.col: float(r.q_BH) for r in T.itertuples()},
    "p": {r.col: float(r.p) for r in T.itertuples()},
    "effect": {r.col: float(r.effect) for r in T.itertuples()},
}

# ---------- Step 4 localise ----------
loc = []
for v in ["carbon_intensity_elec", "per_capita_electricity", "gdp_pc_latest", "net_elec_imports_share_demand"]:
    d = x[["archetype", v]].dropna()
    dn = sp.posthoc_dunn(d, val_col=v, group_col="archetype", p_adjust="holm")
    for a_, b_ in itertools.combinations(ORDER, 2):
        loc.append(
            dict(
                variable=v,
                a=a_,
                b=b_,
                p_holm=dn.loc[a_, b_],
                med_a=d[d.archetype == a_][v].median(),
                med_b=d[d.archetype == b_][v].median(),
            )
        )
loc = pd.DataFrame(loc)
loc.to_csv(f"{O}/posthoc_dunn.csv", index=False)
print(loc.round(4).to_string())
# One-vs-rest contrast for the clustering inputs: rank-biserial correlation only (descriptive, no p-values).
ovr = []
for v in N:
    for a_ in ORDER:
        g1 = x[x.archetype == a_][v]
        g0 = x[x.archetype != a_][v]
        U_, _ = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        rb = 2 * U_ / (len(g1) * len(g0)) - 1
        ovr.append(dict(variable=v, archetype=a_, rank_biserial=rb, median_in=g1.median(), median_out=g0.median()))
ovr = pd.DataFrame(ovr)
ovr.to_csv(f"{O}/one_vs_rest_descriptive.csv", index=False)
print(ovr.pivot(index="variable", columns="archetype", values="rank_biserial").loc[N, ORDER].round(2))

# ---------- Step 5 surrogates ----------
y = x.archetype.values
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Decision tree (5 leaves)": DecisionTreeClassifier(max_leaf_nodes=5, min_samples_leaf=5, random_state=0),
    "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0),
    "Random forest (500 trees)": RandomForestClassifier(500, random_state=0, n_jobs=-1),
    "Multinomial logistic (L2, C=1, z-scored)": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
}
sr = []
for nm, m in models.items():
    s = cross_validate(m, X, y, cv=cv, scoring=["balanced_accuracy", "f1_macro", "accuracy"])
    sr.append(
        dict(
            model=nm,
            bal_acc=s["test_balanced_accuracy"].mean(),
            bal_acc_sd=s["test_balanced_accuracy"].std(),
            f1_macro=s["test_f1_macro"].mean(),
            f1_sd=s["test_f1_macro"].std(),
            acc=s["test_accuracy"].mean(),
        )
    )
sr = pd.DataFrame(sr)
sr.to_csv(f"{O}/surrogates_cv.csv", index=False)
print(sr.round(3).to_string())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
tree = DecisionTreeClassifier(max_leaf_nodes=5, min_samples_leaf=5, random_state=0).fit(Xtr, ytr)
print(export_text(tree, feature_names=N, decimals=1))
print("tree hold-out bal acc:", balanced_accuracy_score(yte, tree.predict(Xte)), "n_test", len(yte))
print(pd.crosstab(pd.Series(yte, name="true"), pd.Series(tree.predict(Xte), name="pred")))
R["tree"] = {
    "text": export_text(tree, feature_names=N, decimals=1),
    "holdout_bal_acc": balanced_accuracy_score(yte, tree.predict(Xte)),
    "n_train": len(ytr),
    "n_test": len(yte),
    "n_leaves": int(tree.get_n_leaves()),
}
t_ = tree.tree_
chain = []
node = 0
while t_.children_left[node] != -1:
    l_, r_ = t_.children_left[node], t_.children_right[node]
    assert t_.children_left[r_] == -1, "expected a chain-shaped tree"
    chain.append(
        dict(
            feature=N[t_.feature[node]],
            threshold=float(t_.threshold[node]),
            leaf=str(tree.classes_[t_.value[r_][0].argmax()]),
            n_train=int(t_.n_node_samples[r_]),
        )
    )
    node = l_
R["tree"]["chain"] = chain
R["tree"]["else_leaf"] = str(tree.classes_[t_.value[node][0].argmax()])
R["tree"]["else_n_train"] = int(t_.n_node_samples[node])
rf = RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pi = permutation_importance(rf, Xte, yte, n_repeats=100, random_state=0, scoring="balanced_accuracy")
PI = pd.DataFrame(
    {
        "feature": N,
        "imp_mean": pi.importances_mean,
        "imp_sd": pi.importances_std,
        "p05": np.percentile(pi.importances, 5, axis=1),
        "p95": np.percentile(pi.importances, 95, axis=1),
    }
).sort_values("imp_mean", ascending=False)
PI.to_csv(f"{O}/rf_perm_importance.csv", index=False)
print(PI.round(3).to_string())
R["rf_holdout_bal_acc"] = balanced_accuracy_score(yte, rf.predict(Xte))
lg = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr, ytr)
co = pd.DataFrame(lg[-1].coef_, index=lg[-1].classes_, columns=N).loc[ORDER]
co.to_csv(f"{O}/logit_coefs.csv")
print("logit coef (log-odds per +1 SD):\n", co.round(2))
R["logit_holdout_bal_acc"] = balanced_accuracy_score(yte, lg.predict(Xte))
# context-only model (non-circular)
d = x.dropna(subset=["income_group"]).copy()
inc_ord = {"Low income": 0, "Lower middle income": 1, "Upper middle income": 2, "High income": 3}
Xc = pd.concat(
    [
        d.income_group.map(inc_ord).rename("income_rank"),
        np.log10(d.per_capita_electricity.clip(lower=1)).rename("log_kwh_pc"),
        np.log10(d.population).rename("log_pop"),
        pd.get_dummies(d.continent, dtype=float),
    ],
    axis=1,
)
ctx = {}
for nm, m in {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "rf_context": RandomForestClassifier(500, random_state=0, n_jobs=-1, min_samples_leaf=3),
}.items():
    s = cross_validate(m, Xc.values, d.archetype.values, cv=cv, scoring=["balanced_accuracy", "accuracy"])
    ctx[nm] = {
        "bal_acc": s["test_balanced_accuracy"].mean(),
        "sd": s["test_balanced_accuracy"].std(),
        "acc": s["test_accuracy"].mean(),
    }
R["context_model"] = {"n": len(d), "features": Xc.columns.tolist(), **ctx}
print("context model:", R["context_model"])

# ---------- RQ3 income vs carbon intensity ----------
IO = ["Low income", "Lower middle income", "Upper middle income", "High income"]
d = x.dropna(subset=["income_group"])
gs = [d[d.income_group == g].carbon_intensity_elec.values for g in IO]
H, p = stats.kruskal(*gs)
dn = sp.posthoc_dunn(d, val_col="carbon_intensity_elec", group_col="income_group", p_adjust="holm")


def boot_rho(a_, b_, B=10000, seed=0):
    rg = np.random.default_rng(seed)
    a_ = np.asarray(a_)
    b_ = np.asarray(b_)
    r = []
    for _ in range(B):
        i = rg.integers(0, len(a_), len(a_))
        r.append(stats.spearmanr(a_[i], b_[i])[0])
    return np.percentile(r, [2.5, 97.5])


g = x.dropna(subset=["gdp_pc_latest"])


def rho(a_, b_):
    r, p_ = stats.spearmanr(a_, b_)
    lo, hi = boot_rho(a_, b_)
    return dict(rho=r, p=p_, ci_lo=lo, ci_hi=hi, n=len(a_))


R["rq3"] = {
    "kw_H": H,
    "kw_p": p,
    "n": len(d),
    "eta2_H": eta2_H(H, 4, len(d)),
    "medians": {g_: float(np.median(v)) for g_, v in zip(IO, gs)},
    "iqr": {g_: [float(np.percentile(v, 25)), float(np.percentile(v, 75))] for g_, v in zip(IO, gs)},
    "ns": {g_: len(v) for g_, v in zip(IO, gs)},
    "dunn_holm": {f"{a_} vs {b_}": float(dn.loc[a_, b_]) for a_, b_ in itertools.combinations(IO, 2)},
    "spearman_gdp_ci": rho(g.gdp_pc_latest, g.carbon_intensity_elec),
    "spearman_gdp_kwh": rho(g.gdp_pc_latest, g.per_capita_electricity),
    "spearman_gdp_lowcarbon": rho(g.gdp_pc_latest, g.low_carbon_share_elec),
    "spearman_incomerank_ci": rho(d.income_group.map(inc_ord), d.carbon_intensity_elec),
}
Hk, pk = stats.kruskal(*[d[d.income_group == g_].per_capita_electricity.values for g_ in IO])
R["rq3"]["kw_kwh"] = {
    "H": Hk,
    "p": pk,
    "eta2_H": eta2_H(Hk, 4, len(d)),
    "medians": d.groupby("income_group").per_capita_electricity.median().loc[IO].to_dict(),
}
print("RQ3", json.dumps(R["rq3"], default=float, indent=0))


# robustness: exclude tiny systems (<1 TWh) where rounding makes carbon intensity noisy
big = x[x.electricity_generation >= 1]
db = big.dropna(subset=["income_group"])
gb = big.dropna(subset=["gdp_pc_latest"])
Hb, pb = stats.kruskal(*[db[db.income_group == g_].carbon_intensity_elec.values for g_ in IO])
R["rq3_robust"] = {
    "n_systems_ge1TWh": len(big),
    "n_lt_1TWh": int((x.electricity_generation < 1).sum()),
    "n_lt_0.1TWh": int((x.electricity_generation < 0.1).sum()),
    "kw_H": Hb,
    "kw_p": pb,
    "n": len(db),
    "eta2_H": eta2_H(Hb, 4, len(db)),
    "spearman_gdp_ci": rho(gb.gdp_pc_latest, gb.carbon_intensity_elec),
}
kb = KMeans(K, n_init=50, random_state=0).fit(big[N].values)
R["rq3_robust"]["ari_clusters_ge1TWh"] = adjusted_rand_score(big.archetype, kb.labels_)
R["rq3_robust"]["sil_ge1TWh"] = silhouette_score(big[N].values, kb.labels_)
print("ROBUST", json.dumps(R["rq3_robust"], default=float))
dr = sp.posthoc_dunn(x, val_col="carbon_intensity_elec", group_col="archetype", p_adjust=None)
print(
    "raw Dunn hydro vs diversified:",
    dr.loc["Hydro-dominated", "Diversified"],
    "MWU:",
    stats.mannwhitneyu(
        x[x.archetype == "Hydro-dominated"].carbon_intensity_elec, x[x.archetype == "Diversified"].carbon_intensity_elec
    ).pvalue,
)
low = x.archetype.isin(["Hydro-dominated", "Diversified"])
R["two_tier"] = {
    "low_median": float(x[low].carbon_intensity_elec.median()),
    "fossil_median": float(x[~low].carbon_intensity_elec.median()),
    "n_low": int(low.sum()),
    "n_fossil": int((~low).sum()),
}
print(R["two_tier"])
print("CI>900:", x[x.carbon_intensity_elec > 900][["country", "carbon_intensity_elec", "electricity_generation"]].values.tolist())
print("neg silhouette:", x[x.sil < 0][["country", "archetype", "sil"]].values.tolist())

# ---------- RQ4 transitions ----------
p00 = panel[panel.year == 2000].set_index("iso_code")
both = x.set_index("iso_code").join(p00[N + ["carbon_intensity_elec", "electricity_generation"]], rsuffix="_2000", how="inner")
lab00 = km.predict(both[[c + "_2000" for c in N]].values)
both["archetype_2000"] = [names[i] for i in lab00]
tm = pd.crosstab(both.archetype_2000, both.archetype).reindex(index=ORDER, columns=ORDER, fill_value=0)
tm.to_csv(f"{O}/transition_matrix.csv")
print(tm)
stay = int(np.trace(tm.values))
R["rq4"] = {"n_both": len(both), "stayed": stay, "moved": len(both) - stay, "stay_pct": 100 * stay / len(both)}
mov = both[both.archetype != both.archetype_2000].sort_values("electricity_generation", ascending=False)
R["rq4"]["largest_movers"] = [f"{r.country}: {r.archetype_2000} → {r.archetype}" for r in mov.head(25).itertuples()]
print(R["rq4"])
both["ci_change"] = both.carbon_intensity_elec - both.carbon_intensity_elec_2000
print(both.groupby("archetype_2000").ci_change.describe().round(0))
R["rq4"]["ci_down_n"] = int((both.ci_change < 0).sum())
R["rq4"]["ci_up_n"] = int((both.ci_change > 0).sum())
R["rq4"]["ci_change_median"] = float(both.ci_change.median())
wl = stats.wilcoxon(both.carbon_intensity_elec, both.carbon_intensity_elec_2000)
R["rq4"]["wilcoxon_p"] = float(wl.pvalue)
top = both.sort_values("electricity_generation", ascending=False).head(20)[
    [
        "country",
        "carbon_intensity_elec_2000",
        "carbon_intensity_elec",
        "ci_change",
        "archetype_2000",
        "archetype",
        "electricity_generation",
    ]
]
top.to_csv(f"{O}/top20_dumbbell.csv")
print(top.round(0).to_string())
R["rq4"]["ci_median_2000"] = float(both.carbon_intensity_elec_2000.median())
R["rq4"]["ci_median_2023"] = float(both.carbon_intensity_elec.median())
R["rq4"]["from_counts"] = both.archetype_2000.value_counts().to_dict()
print("RQ4b", {k_: v_ for k_, v_ in R["rq4"].items() if k_ != "largest_movers"})
R["rq4"]["top20_gen_share"] = float(100 * top.electricity_generation.sum() / x.electricity_generation.sum())
both_out = both.reset_index()
both_out.to_csv(f"{O}/both_2000_2023.csv", index=False)
EU = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
]
assert set(EU) <= set(x.country), set(EU) - set(x.country)
eu = x[x.country.isin(EU)].sort_values("electricity_generation", ascending=False)
print(eu[["country", "electricity_generation", "carbon_intensity_elec", "archetype"]].head(10).round(0).to_string())
print("EU archetypes:", eu.archetype.value_counts().to_dict())
R["eu"] = {"top6": eu.country.head(6).tolist(), "archetypes": eu.archetype.value_counts().to_dict()}
it = panel[panel.country == "Italy"].set_index("year")
R["italy"] = {
    "ci_2000": it.loc[2000, "carbon_intensity_elec"],
    "ci_2023": it.loc[2023, "carbon_intensity_elec"],
    "ci_2024": it.loc[2024, "carbon_intensity_elec"],
    "mix_2000": it.loc[2000, N].round(1).to_dict(),
    "mix_2023": it.loc[2023, N].round(1).to_dict(),
    "archetype_2000": both.set_index("country").loc["Italy", "archetype_2000"],
    "archetype_2023": both.set_index("country").loc["Italy", "archetype"],
}
print(R["italy"])
# ---------- sample-size chain (every n used in the talk) ----------
all_sys = panel.drop_duplicates("iso_code").set_index("iso_code").country
not23 = sorted(all_sys[~all_sys.index.isin(x.iso_code)].tolist())
not00 = sorted(x[~x.iso_code.isin(both.index)].country.tolist())
SS = pd.DataFrame(
    [
        dict(sample="Clean panel 2000-2024", n=int(panel.iso_code.nunique()), excluded="-", used_for="trends (F02, F15)"),
        dict(
            sample="2023 snapshot",
            n=len(x),
            excluded="no usable 2023 data: " + ", ".join(not23),
            used_for="PCA, UMAP, clustering, external tests (F03-F09, F11, F12)",
        ),
        dict(
            sample="In both 2000 and 2023",
            n=len(both),
            excluded="series start after 2000: " + ", ".join(not00),
            used_for="transitions (F13, F14)",
        ),
        dict(
            sample="With World Bank income group",
            n=int(x.income_group.notna().sum()),
            excluded=f"{int(x.income_group.isna().sum())} territories without a WB class: "
            + ", ".join(sorted(x[x.income_group.isna()].country)),
            used_for="income tests (F10, F11)",
        ),
        dict(
            sample="With GDP per capita (2022)",
            n=int(x.gdp_pc_latest.notna().sum()),
            excluded=f"{int(x.gdp_pc_latest.isna().sum())} systems not in the Maddison database",
            used_for="GDP correlations and test (F10, F11)",
        ),
    ]
)
SS.to_csv(f"{O}/sample_sizes.csv", index=False)
print(SS.to_string())
R["sample_sizes"] = SS.set_index("sample").n.to_dict()
x.to_csv(f"{O}/xs2023_labeled.csv", index=False)
json.dump(R, open(f"{O}/results.json", "w"), default=float, indent=1)
