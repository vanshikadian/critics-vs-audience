# Split Decision

Split Decision is an interactive data story for movie fans who want to look beyond a single approval score. It asks how much critics and audiences differ across titles and genres, and whether genre patterns change when the sample uses higher review-count minimums or the same mix of release periods.

With the default minimums of 20 critic reviews and 100 audience ratings, 10,503 titles qualify. Their overall mean scores are close, but the mean absolute gap is 15.19 percentage points. Horror's mean changes from −3.43 to +1.33 points under the stricter 50/1,000 rule, while its median stays negative. Animation's mean remains close to zero under both rules.

## Data and methods

The source is [Stefano Leone's Rotten Tomatoes dataset on Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset), collected through October 31, 2020. The publisher lists CC0. The movie table contains 17,712 titles and 22 original columns, including some television and other non-theatrical entries. The separate critic-review text is not used.

Each source ID identifies a title. Preparation retains every source row, leaves missing scores and counts unfilled, and keeps valid zero scores. Derived fields include the audience-minus-critic gap, its absolute size, release periods and eligibility flags. Review counts filter titles; every included title has equal weight. The notebook explains the quality checks and decisions.

The scores are approval percentages, not average stars. Genres overlap, the catalog is a selected historical snapshot, and the two scores represent different populations and scoring rules. These descriptive comparisons cannot explain individual opinions or establish causes. The cutoffs and near-zero tolerance are analytical choices, not confidence levels. No predictive model is needed for these questions.

## Run the app

Use Python 3.11. From the repository folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The included `data/processed/movie_scores.csv` and `preparation.json` let the app run without the raw CSV or an API key. The five tabs cover overall gaps, genres, a comparison of two count filters, release periods, and data/methods. The sidebar sets the main views; Filter test uses its own A/B count presets. Tables can be downloaded as CSV files.

## Reproduce the analysis

Download `rotten_tomatoes_movies.csv` from the linked source into `data/raw/`. The notebook and full data checks require that raw file. Then run:

```bash
python scripts/prepare_data.py
python scripts/check_project.py
python scripts/check_app.py
```

Open `notebooks/01_data_and_eda.ipynb` in a Jupyter-capable editor using the same environment and run all cells. Saved outputs include the figures and interpretations. Preparation preserves the raw file and records source/output fingerprints. The checks cover formulas, data consistency, filter behavior, period calculations, empty results and CSV contents. Generated check outputs and local backups are ignored by Git.

## AI assistance

OpenAI Codex (GPT-6.1) supported app development, including code, the Streamlit interface, and deployment setup. I used it to explore implementation options and troubleshoot issues. I chose the project direction, reviewed and edited the generated material, and checked results against the underlying data and recalculations, including how review-count filters affected score comparisons. I developed the analytical rationale and final interpretations and the presentation.
