# Five ways to keep the lights on

**How 211 countries make their electricity, 2000–2024**

## What this project is about

Every country makes its electricity from a different mix of coal, gas, oil, hydro, nuclear, wind, solar and other sources. We asked whether those mixes fall into a few clear types, what defines each type, and how this relates to wealth and to CO₂ emissions.

Using the Our World in Data energy dataset (Ember and Energy Institute), we clean 25 years of data for 211 countries, group countries by their 2023 electricity mix with k-means clustering, and explain the groups with PCA, UMAP, decision trees, random forests and hypothesis tests.

## Four questions, four answers

1. **Is the world's electricity getting cleaner?** Only per unit. Each kWh is 11% cleaner than in 2000, but the world makes twice as much electricity, so power-sector emissions are up 81%.
2. **Do countries fall into a few types?** Yes, five: hydro-dominated, diversified, gas-dominated, oil-dependent and coal-heavy. Four simple rules (oil, gas, hydro, coal share) sort countries into them with 95% accuracy. Coal-heavy countries are only 12% of countries but make half of the world's electricity.
3. **Do richer countries have cleaner electricity?** No. Richer countries use far more electricity per person, but it is not cleaner per kWh.
4. **Who changed type since 2000, and did Italy?** Most countries (149 of 208) kept their type. Italy's electricity is 44% cleaner than in 2000, but it is still gas-dominated.

## What's in this repository

- **Presentation:** [`github_repo/presentation/`](github_repo/presentation/) 
- **Code:** [`github_repo/scripts/`](github_repo/scripts/), run in the order given in [`github_repo/README.md`](github_repo/README.md)
- **Results and figures:** [`github_repo/outputs/`](github_repo/outputs/)
- **Methods and design choices:** [`github_repo/METHODOLOGY_REPORT.md`](github_repo/METHODOLOGY_REPORT.md)
