# Split Decision

Critics and audiences can give very different scores to the same movie. I’m looking at how often that happens and whether the gap varies across genres, using Stefano Leone’s [Rotten Tomatoes dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset).

So far, the project includes data checks, two comparisons in a notebook, and a Streamlit app for exploring the results by genre and review count. Next I’ll compare release periods and check which findings hold up under stricter filters, then write up the analysis for the presentation.

The data are a snapshot ending in October 2020. The scores measure approval, not average stars, and they cannot tell us why someone liked a movie.

## Running it

Use Python 3.11. Download `rotten_tomatoes_movies.csv` from the dataset above and put it in `data/raw/`. The CSV is already included in my local folder but is excluded from Git.

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

AI assistance disclosure: Codex supported the streamlit app code, checks and UI.
