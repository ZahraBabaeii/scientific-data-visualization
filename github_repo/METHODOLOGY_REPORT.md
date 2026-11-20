# Methodology report
**Five ways to keep the lights on: electricity-mix archetypes of 211 national power systems, 2000–2024**
Scientific and Data Visualization — 2026/2027, Università degli Studi di Milano. Authors: `[Group Member Names]`

Every number below is produced by `scripts/` and stored in `outputs/results.json` or the CSV named next to it. `scripts/verify.py` recomputes the headline numbers from the raw file (46/46 checks pass).

---

## 1. Goal and research questions

**Goal.** Describe how countries generate electricity; check whether a small number of mix "archetypes" exists; and test whether wealth explains which countries have clean power.

| | Domain question | Mode (Keim / Nagel) |
|---|---|---|
| RQ1 | Is world electricity getting cleaner? | Presentation |
| RQ2 | Do national power systems fall into a few mix types, and what defines them? | Exploratory |
| RQ3 | **H1 (stated before testing):** richer countries have lower carbon intensity of electricity. | Confirmatory |
| RQ4 | Who changed type between 2000 and 2023, and where does Italy stand? | Presentation |

### Task abstraction (Munzner's nested model, level 2)

| | Abstract task | Data types | Encoding chosen (level 3) | Figure |
|---|---|---|---|---|
| RQ1 | Track trends; compare 3 series of different units | Temporal × quantitative | Lines indexed to 2000 on one axis (no dual axis); slopegraph for shares | F02 |
| RQ2 | Identify clusters; summarise each; compare their weight | 9 quantitative shares (compositional, sum 100) → 1 categorical label | 2-D position (PCA/UMAP); heatmap of means; aligned bars; equal-area map | F03–F08 |
| RQ2 validation | Compare distributions across groups; rank associations | Categorical × quantitative / categorical | Notched box plots + jitter; dot plot of effect sizes | F09, F11 |
| RQ2 explanation | Derive rules separating groups | Quantitative → categorical | Tree diagram; interval dot plot; coefficient heatmap | F12 |
| RQ3 | Compare distributions over ordered groups; correlate | Ordinal × quantitative; quantitative × quantitative | Notched box plots + jitter; Spearman ρ with bootstrap CI | F10 |
| RQ4 | Compare two time points; track trends for a few items | Categorical × temporal; quantitative × temporal | Count matrix; dumbbell; small multiples with shared axes | F13–F15 |

---

## 2. Data and provenance

- **Primary:** Our World in Data energy dataset (`owid-energy-data.csv`, pinned commit in `download_data.py`). 23,377 rows × 130 columns, 314 entities, 1900–2025. The electricity columns come from Ember's yearly electricity data. **Carbon intensity is modelled**: Ember multiplies generation by IPCC life-cycle emission factors. Our "Oil" column is Ember's *other fossil* category (oil and petroleum products, manufactured gases, waste), which is why oil-shale Estonia is oil-dependent.
- **Enrichment:** World Bank income groups (2020 vintage); OWID continents; UN M49 regions; Maddison GDP per capita (inside the OWID file, latest year 2022); Natural Earth 1:110m boundaries (map only).

### Smell test (`prep.py`, `audit*.py`)
- **Aggregates mixed with countries:** 93 entities without an ISO code (World, "Europe (EI)", OECD, income groups, USSR…) sit in the same rows as countries.
- **Coverage step:** 76–81 countries report generation before 2000, then 212–214 from 2000 to 2023, 197 in 2024 and 91 in 2025 (partial year).
- **Hidden artefact, structural zeros:** "no plant of this type" is stored as missing, not 0. A plain `dropna()` on the nine shares would delete 29 systems carrying **64.9% of 2023 world generation** (China, India, Japan, Russia, Brazil…). In each of the 820 affected country-years the observed shares already sum to 100% (|residual| < 0.05 pp), so the missing shares are set to 0 (1,435 cells).
- **Other findings:** Kosovo has data but no ISO code; Lesotho and Niue report 0 TWh; Ukraine has no 2023–24 data; three micro-systems show exactly 1,000 gCO₂e/kWh (rounding); no duplicate (country, year) pairs.

### Cleaning and transformation log (`outputs/cleaning_log.csv`)

| # | Action | Reason | Rows before → after |
|---|---|---|---|
| 1 | Drop 93 entities without ISO code; keep "World" separately for RQ1; recode Kosovo to `OWID_KOS` and keep it | Aggregates double-count | 23,377 → 17,291 |
| 2 | Keep 22 of 130 columns | The rest are primary-energy metrics outside the question | 17,291 → 17,291 |
| 3 | Keep years 2000–2024 | Coverage step in 2000; 2025 partial | 17,291 → 5,484 |
| 4 | Drop country-years with missing generation | Nothing to analyse | 5,484 → 5,322 |
| 5 | Drop country-years with 0 TWh generation | Shares undefined (0/0) | 5,322 → 5,296 |
| 6 | Missing share → 0 where the other shares sum to 100% | Structural zeros (820 country-years, 1,435 cells) | 5,296 → 5,296 |
| 7 | Left-join income group, continent, UN region on ISO-3 | Enrichment | 5,296 → 5,296 |
| 8 | Derive GDP per capita = GDP / population, latest year ≤ 2022, carried forward | External variable; no GDP after 2022 | 5,296 → 5,296 |

### Sample sizes (`outputs/sample_sizes.csv`)

| Sample | n | Excluded | Used for |
|---|---|---|---|
| Clean panel 2000–2024 | 214 systems | – | Trends (F02, F15) |
| 2023 snapshot | 211 | Lesotho, Ukraine, Western Sahara (no usable 2023 data) | PCA, UMAP, clustering, external tests |
| In both 2000 and 2023 | 208 | East Timor, Montenegro, South Sudan (series start after 2000) | Transitions (F13, F14) |
| With World Bank income group | 202 | 9 territories without a WB class | Income tests (F10, F11) |
| With GDP per capita | 163 | 48 systems not in the Maddison database | GDP test and correlations |

---

## 3. Statistical and modelling pipeline (`analysis.py`)

**Features.** The nine 2023 generation shares (coal, gas, oil, nuclear, hydro, wind, solar, bioenergy, other renewables), in % of generation. All share one unit, so they are **centred but not z-scored**. Z-scoring would give "other renewables" (mean < 1%) the same weight as hydro. Z-scored k-means is reported as a sensitivity check.

**Dimensionality reduction (Lecture 5).**
- PCA: PC1 46.1%, PC2 30.2%, PC3 13.9%, PC4 4.9% (PC1–3 = 90.1%). On z-scored shares the variance is flat (18.6%, 17.2%, 15.4%…), which confirms that z-scoring would dilute the structure.
- UMAP: `n_neighbors=15, min_dist=0.1, metric="euclidean", init=PCA(2), random_state=0`. Twenty runs (seeds 0–4 × {10, 15, 30} neighbours × min_dist {0.1, 0.3}) give trustworthiness (k = 10) of 0.975–0.980, against 0.934 for PCA's first two components. The embedding is always shown uncoloured beside the coloured version, and no claim relies on UMAP distances or group sizes.

**Clustering (Lectures 6a–6b).**
- k-means, Euclidean, `n_init=100, random_state=0`, on the 211 × 9 share matrix.
- Choice of k (`k_selection.csv`): silhouette 0.484 (k = 4), **0.485 (k = 5)**, 0.452 (k = 6). Bootstrap ARI over 200 resamples (mean; 5th percentile): 0.93; 0.88 (k = 4), **0.92; 0.84 (k = 5)**, 0.83; 0.70 (k = 6).
- **k − 1 (`k_sensitivity_k4.csv`):** all 25 coal-heavy and 25 of 29 diversified systems merge into one 52-system cluster (mean coal 32%, hydro 14%). Sizes 49 / 50 / 52 / 60. ARI vs k = 5: 0.86. We keep k = 5 because the merge mixes the dirtiest and the cleanest non-hydro systems.
- **k + 1 (`k_sensitivity_k6.csv`):** the oil group splits into "pure oil" (51 systems, mean oil 89%) and a 22-system oil–hydro mixed group (10 from oil, 5 hydro, 5 diversified, 2 coal). Coal shrinks to 21. Sizes 21 / 22 / 27 / 45 / 45 / 51. ARI vs k = 5: 0.83. The new group is a blend, not a new archetype.
- **Second method (`agreement_*.csv`):**
  - Agglomerative clustering, Ward linkage, k = 5: **ARI 0.85** with k-means. The oil cluster matches exactly (61/61); 22 of 29 diversified and 22 of 25 coal-heavy systems agree.
  - Average linkage: ARI 0.81. Complete linkage: 0.77.
  - Gaussian mixture on PC1–PC4 (k = 5, `n_init=20, reg_covar=1e-3`): tied covariance ARI 0.93; full covariance ARI 0.41. Full covariance estimates a separate 4 × 4 covariance matrix for each of five components (clusters of 25–61 points). It then groups systems by the shape of their spread rather than their location: one component mixes 27 gas-dominated with 24 hydro-dominated systems. We read this as over-flexibility, not competing structure, and report it as a disagreement.
- Other sensitivity checks: Hellinger distance (√share) ARI 0.84; z-scored features ARI 0.83; excluding the 53 systems below 1 TWh, ARI 0.96.
- **Per-cluster silhouette:** oil 0.68, gas 0.50, hydro 0.47, coal 0.40, diversified 0.17 (the weakest; 4 systems have negative silhouette).

**Six-step cluster explanation.**
1. **Summarise:** `cluster_summary.csv`, `cluster_means.csv`, `cluster_medians.csv`.
2. **Test, external variables only.** The nine shares built the clusters, so testing them would be circular (Lecture 6b). Their η² (0.09–0.68) is used **only to order the heatmap columns in F06** and is labelled descriptive; no p-values are reported for them. The test family is 8 variables not used for clustering: Kruskal–Wallis for six numeric variables; χ² with a Monte-Carlo p-value (20,000 permutations, because some expected counts are < 5) for two categorical ones.
3. **Adjust:** Benjamini–Hochberg FDR across **m = 8** tests.
4. **Localise:** Dunn tests with Holm correction within each variable (`posthoc_dunn.csv`).
5. **Surrogate models:** see below.
6. **Show:** one colour per archetype in every figure.

**External tests** (`tests_external.csv`, BH over 8):

| Variable | Test | Statistic | n | q (BH) | Effect size |
|---|---|---|---|---|---|
| Carbon intensity † *(derived from the mix)* | Kruskal–Wallis | H(4) = 142.9 | 211 | < 0.001 | η² = 0.67 |
| System size (TWh) | Kruskal–Wallis | H(4) = 106.4 | 211 | < 0.001 | η² = 0.50 |
| Population | Kruskal–Wallis | H(4) = 62.7 | 211 | < 0.001 | η² = 0.29 |
| GDP per capita (2022) | Kruskal–Wallis | H(4) = 38.1 | 163 | < 0.001 | η² = 0.22 |
| Electricity per person | Kruskal–Wallis | H(4) = 19.9 | 211 | < 0.001 | η² = 0.08 |
| Net imports (% demand) | Kruskal–Wallis | H(4) = 1.35 | 211 | 0.853 | η² ≈ 0 |
| Continent | χ² Monte-Carlo | χ²(20) = 117 | 211 | < 0.001 | V = 0.37 |
| Income group | χ² Monte-Carlo | χ²(12) = 51.2 | 202 | < 0.001 | V = 0.29 |

η²_H = (H − k + 1)/(n − k). † Carbon intensity is computed from the generation mix, so its large effect is expected and is **not** independent validation; it is reported and drawn separately (grey diamond in F11). Dunn–Holm on carbon intensity shows two tiers: hydro vs diversified p = 1.00; gas vs oil vs coal p = 1.00; every low-carbon vs fossil pair p < 0.001.

**Surrogate models** (step 5; predict the archetype from the nine shares; `surrogates_cv.csv`). Evaluation: 10 × repeated stratified 5-fold CV, balanced accuracy, majority baseline 0.20.

| Model | Settings | Balanced accuracy (mean ± SD) |
|---|---|---|
| Decision tree | `max_leaf_nodes=5, min_samples_leaf=5` | 0.95 ± 0.03 |
| Random forest | 500 trees, permutation importance on 30% hold-out (100 repeats) | 0.96 ± 0.03 |
| Multinomial logistic | L2, C = 1, z-scored inputs | 0.95 ± 0.04 |

- The tree, fitted on a 70% split (n = 147; hold-out n = 64, balanced accuracy 0.95), reads: Oil > 47% → oil-dependent; else Gas > 44% → gas-dominated; else Hydro > 39% → hydro-dominated; else Coal > 30% → coal-heavy; else diversified.
- Random-forest permutation importance: oil, gas, hydro and coal only.
- **Non-circular check:** income rank, continent, log population and log kWh per person alone predict the archetype with balanced accuracy 0.47 ± 0.07 (n = 202). Context explains part of the mix, but far from all of it.

**RQ3 (confirmatory, single pre-specified hypothesis, unadjusted).**
- Carbon intensity by income group (n = 202): Kruskal–Wallis H(3) = 6.50, p = 0.090, η² = 0.02. Medians 350 / 500 / 566 / 437 gCO₂e/kWh (low → high income).
- Spearman ρ(GDP per capita, carbon intensity) = −0.01, 95% bootstrap CI −0.17 to 0.15 (10,000 resamples, n = 163).
- Contrast: kWh per person by income group, H(3) = 144.6, η² = 0.72; ρ(GDP, kWh per person) = 0.91.
- Robustness, excluding systems < 1 TWh (n = 155): H(3) = 8.67, p = 0.034, η² = 0.04, ρ = −0.02. The effect stays small in every version.

**RQ4.** Each system's 2000 mix is assigned to the nearest 2023 centroid, so the categories mean the same in both years. 149 of 208 systems (72%) keep their archetype. Coal → diversified: 10 (US, Germany, Spain, South Korea…); oil → gas: 13. Carbon intensity fell in 133 systems and rose in 71 (Wilcoxon signed-rank p < 0.001). Italy: 502 → 281 gCO₂e/kWh (−44%), oil 34% → 4%, wind + solar 0% → 22%, still gas-dominated.

---

## 4. Design rules applied to every figure

- **Channels:** quantities that carry the argument use position or length on a common, zero-based scale. Hue is categorical only. Lightness is used once for a magnitude (F13 counts, which are also printed).
- **Palette:** archetype colours from the Okabe–Ito colour-blind-safe set: hydro `#0072B2`, diversified `#009E73`, gas `#E69F00`, oil `#D55E00`, coal `#882255` (wine replaces Okabe–Ito's pink to keep the minimum CVD ΔE ≥ 11). Checked with a CVD validator. Grey is used for context. The diverging map `PuOr` (ColorBrewer) is used only where 0 is meaningful (z-scores; logistic coefficients).
- **Integrity:** bars start at 0 (lie factor 1.0); no 3-D; no dual axes; modelled values marked †; derived 2000 archetypes stated in the caption.
- **Stranger test:** every figure has a takeaway title, a subtitle defining marks, units, n and the test, and a source line. Labels are direct wherever possible.

### Four-line figure specifications

| Fig | (1) Question | (2) Encoding | (3) Colour semantics | (4) Why this chart |
|---|---|---|---|---|
| F01 | Which years can we trust? | Bar length = countries reporting, per year | Blue = kept, grey = dropped | A count over time; length on a zero baseline shows the 2000 step at a glance |
| F02 | Is electricity cleaner per kWh and in total? | (a) lines indexed to 2000; (b) slopegraph of shares 2000 → 2024 | Ink, vermilion, blue for the three series; archetype hues for highlighted sources, grey for the rest | Indexing puts three units on one axis without a dual axis; slopes compare 9 shares at two dates |
| F03 | How many dimensions matter? | (a) bar length = % variance; (b) 2-D position + loading arrows | Blue = components discussed | Scree justifies 2-D views; biplot shows which sources drive the axes |
| F04 | Is the structure real or painted on? | 2-D position, PCA and UMAP, grey and coloured | Archetype hue only on the right | The mandatory uncoloured check (Lecture 5) |
| F04b | Is UMAP stable? | Six small multiples with identical encoding | Archetype hue | Varies seed and hyper-parameters in one view |
| F05 | Why k = 5? | Position: silhouette and bootstrap ARI vs k; band = 5th–95th percentile | Blue = chosen k | Shows k ± 1 in both criteria |
| F06 | What defines each archetype? | Cell text = mean share; cell colour = z-score vs all systems | Diverging PuOr, 0 = average; colour chip = archetype | Exact values printed, colour only as context; columns ordered by descriptive η² |
| F07 | Do country counts mislead? | Bar length, three aligned panels, shared 0–60% scale | Archetype hue | Same scale makes "29% of systems vs 0.4% of electricity" visible |
| F08 | Where are the archetypes? | Equal Earth map, fill = archetype | 5 categorical hues + grey for no data | The question is spatial; equal-area avoids Mercator inflation |
| F09 | Do archetypes differ in carbon intensity? | Notched box + jitter; y from 0 | Archetype hue | Notch = 95% CI of the median; distribution and n visible |
| F10 | H1: does income buy cleaner power? | Notched box + jitter, two panels (linear; log for kWh) | Grey boxes; dots in archetype hue (a) | Shows the null beside a strong effect in the same layout |
| F11 | What separates the archetypes, independently of their inputs? | Dot position = effect size; filled/hollow = q < 0.05 | Blue = independent and significant; grey diamond = derived | Effect sizes with q replace significance stars |
| F12 | Can simple rules reproduce the clusters? | Tree diagram; interval dots; coefficient heatmap | Archetype hue for leaves; PuOr for coefficients | Three surrogate families, one each for rules, importance and direction |
| F13 | Who changed archetype? | Matrix of printed counts; outlined diagonal | Grey lightness repeats the count; diagonal outline in archetype hue | 25 exact counts stay readable, unlike crossing Sankey ribbons |
| F14 | Did the largest systems get cleaner? | Dumbbell, common x-scale from 0 | Hollow = 2000; filled = 2023 in archetype hue | Shows direction and size of change per country |
| F15 | Where is Italy among large EU systems? | Small multiples, shared axes, peers in grey | Line in archetype hue | Direct comparison without a legend |

---

## 5. Limitations and trade-offs

1. **Associational, not causal.** Surrogate models describe cluster boundaries; external tests show association only.
2. **Carbon intensity is modelled** (IPCC factors × generation), so it is partly determined by the clustering inputs.
3. **One cross-section (2023), unweighted.** Nauru counts as much as China. F07 shows the weighting effect; generation-weighted clustering is future work.
4. **k-means assumptions.** k-means assumes compact, roughly spherical clusters of similar spread. The diversified cluster is a residual category (silhouette 0.17). Full-covariance GMM disagrees (ARI 0.41).
5. **Compositional data.** Shares sum to 100%. Euclidean distance on raw shares is a pragmatic choice; Hellinger gives ARI 0.84. Log-ratio (Aitchison) distances are future work, as zeros need treatment first.
6. **Coverage.** Income classes are the 2020 vintage; GDP stops in 2022; 48 systems lack GDP. The 1:110m map cannot show 41 small systems (33 oil-dependent islands).
7. **Frozen data release.** Analysis uses a pinned OWID commit. Later releases revise Ember series slightly.
8. **"Oil" means Ember's other-fossil category,** which also includes manufactured gases and waste.

**Future work:** generation-weighted and trajectory-based clustering (2000–2024 paths); log-ratio distances with zero replacement; monthly Ember data for seasonal mixes; adding capacity and price data to explain transitions.
