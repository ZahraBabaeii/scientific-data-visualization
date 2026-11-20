# Five ways to keep the lights on
### Electricity-mix archetypes of 211 national power systems, 2000–2024

Group project for **Scientific and Data Visualization — 2026/2027**, Università degli Studi di Milano (AnacletoLab).
Authors: `[Group Member Names]`

We use the Our World in Data energy dataset (electricity data from Ember) to ask four questions:

| | Question | Mode |
|---|---|---|
| RQ1 | Is world electricity getting cleaner? | Presentation |
| RQ2 | Do national power systems fall into a few mix "archetypes"? | Exploratory |
| RQ3 | H1: richer countries have cleaner electricity. | Confirmatory |
| RQ4 | Who changed archetype between 2000 and 2023, and where is Italy? | Presentation |

**Headline results**
- World generation +102%, carbon intensity −11%, power-sector emissions +81% (2000–2024).
- k-means (k = 5) on the nine generation shares gives five archetypes: hydro-dominated (50 systems), diversified (29), gas-dominated (46), oil-dependent (61) and coal-heavy (25). Silhouette 0.485; bootstrap ARI 0.92; Ward agreement ARI 0.85.
- H1 is not supported. Carbon intensity does not differ by World Bank income group (Kruskal–Wallis H(3) = 6.50, p = 0.090, η² = 0.02). Electricity per person does (H(3) = 144.6, η² = 0.72).
- 149 of 208 systems (72%) are in the same archetype as in 2000. Italy cut its carbon intensity by 44% while staying gas-dominated.

The full method, cleaning log, test tables and figure specifications are in [`METHODOLOGY_REPORT.md`](METHODOLOGY_REPORT.md).

---

## Reproduce

Python 3.13 (3.11+ should work).

```bash
pip install -r requirements.txt
python scripts/download_data.py    # step 0: raw inputs -> data/ (pinned commits, ~10 MB)
python scripts/prep.py             # step 1: audit + cleaning -> outputs/cleaning_log.csv, panel_clean.csv, xs2023.csv
python scripts/analysis.py         # step 2: PCA, k-means, k ± 1, Ward/GMM, UMAP checks, tests, surrogate models, RQ3, RQ4
python scripts/umap_grid.py        # step 3: six UMAP runs for the stability figure
python scripts/figs.py             # step 4: figures with titles -> outputs/figs/
python scripts/figs.py --slides    # step 4b: same figures without the title line (the slide title carries it) -> outputs/figs_slides/
python scripts/verify.py           # step 5: recomputes every headline number from the raw file; prints PASS/FAIL
```

All randomness is seeded: k-means `random_state=0`, bootstrap seed 42, UMAP seed 0, train/test split seed 0. On the pinned data, `verify.py` reports **46 of 46 checks passed**. UMAP coordinates can differ slightly across numba/CPU builds; no reported number depends on them except the trustworthiness scores.

If GitHub rate-limits the download (HTTP 429), wait a minute and re-run `download_data.py`. It skips files that are already present.

## Data sources

| File (under `data/`) | Source | Licence |
|---|---|---|
| `owid_energy/owid-energy-data.csv`, `owid-energy-codebook.csv` | [owid/energy-data](https://github.com/owid/energy-data) (electricity: Ember Yearly Electricity Data; energy: Energy Institute Statistical Review; GDP: Maddison Project; population: OWID) | CC BY 4.0 |
| `other/income_groups.csv` | World Bank income classification (2020 vintage), via [owid/covid-19-data](https://github.com/owid/covid-19-data/tree/master/scripts/input/wb) | CC BY 4.0 |
| `other/continents.csv` | OWID continent mapping, via [owid/covid-19-data](https://github.com/owid/covid-19-data/tree/master/scripts/input/owid) | CC BY 4.0 |
| `other/country-codes.csv` | UN M49 regions, via [datasets/country-codes](https://github.com/datasets/country-codes) | ODC-PDDL |
| `ne/ne_110m_admin_0_countries.geojson` | [Natural Earth](https://github.com/nvkelso/natural-earth-vector) 1:110m boundaries (map only) | Public domain |

`download_data.py` pins every file to the commit we analysed. Set `PIN = False` in the script to fetch the latest versions instead; the numbers will then change slightly. Raw data is not committed (see `.gitignore`).

## Repository structure

```
├── README.md
├── METHODOLOGY_REPORT.md         written reference: goal, task abstraction, cleaning log, statistics, figure specs, limitations
├── requirements.txt
├── LICENSE
├── scripts/
│   ├── download_data.py          step 0  fetch raw inputs (pinned)
│   ├── prep.py                   step 1  smell test, cleaning, structural zeros, joins
│   ├── analysis.py               step 2  all statistics and models -> outputs/results.json + CSV tables
│   ├── umap_grid.py              step 3  UMAP seeds / hyper-parameters
│   ├── figs.py                   step 4  all figures (colour-blind-safe palette, shared across figures)
│   └── verify.py                 step 5  independent re-computation of headline numbers
├── data/                         raw inputs (git-ignored; created by download_data.py)
└── outputs/
    ├── cleaning_log.csv          every filter with rows before/after
    ├── sample_sizes.csv          214 -> 211 -> 208 / 202 / 163, with the excluded systems named
    ├── results.json              every number quoted in the slides
    ├── summary_tables.xlsx       the main tables in one workbook
    ├── tests_external.csv        8 external tests, BH-adjusted q
    ├── k_selection.csv, k_sensitivity_k4.csv, k_sensitivity_k6.csv, agreement_*.csv
    ├── ... one CSV per further table (centroids, profiles, post-hoc tests, surrogates, transitions)
    ├── logs/                     console output of each step (verify.log = the 46 checks)
    ├── figs/                     final figures (PNG, 200 dpi)
    └── figs_slides/              the same figures without the title line, as used in the slides
```

## Figures

| File | Shows |
|---|---|
| F01_coverage | Countries reporting electricity per year: why 2000–2024 |
| F02_world_trend | World generation, emissions and carbon intensity indexed to 2000; source shares 2000 vs 2024 |
| F03_pca | Scree plot and PC1–PC2 biplot |
| F04_embeddings | PCA and UMAP, uncoloured and coloured by archetype |
| F04b_umap_stability | Six UMAP runs (seeds, `n_neighbors`, `min_dist`) |
| F05_choose_k | Silhouette and bootstrap ARI for k = 2…10 |
| F06_profiles | Mean share per archetype (z-score heatmap; descriptive only) |
| F07_weight | Share of systems vs people vs electricity per archetype |
| F08_map | Archetype map (Equal Earth projection) |
| F09_ci_by_archetype | Notched box plots of carbon intensity by archetype (derived variable) |
| F10_income | RQ3: carbon intensity and kWh per person by income group (notched) |
| F11_effects | Effect sizes for the 8 external variables, BH over 8 tests |
| F12_surrogates | Decision tree, random-forest permutation importance, penalised logistic coefficients |
| F13_transitions | Archetype in 2000 vs 2023 (counts) |
| F14_dumbbell | Carbon intensity 2000 vs 2023, 20 largest systems |
| F15_eu | Carbon intensity 2000–2024, six largest EU systems (small multiples) |

## Licence

Code: MIT. Derived data and figures: CC BY 4.0, with attribution to Our World in Data, Ember, the Energy Institute, the World Bank, the Maddison Project and Natural Earth.
