"""Movie-level comparisons for the October 2020 Rotten Tomatoes snapshot."""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/rotten_tomatoes_movies.csv"
SCORES = ["tomatometer_rating", "audience_rating"]
COUNTS = ["tomatometer_count", "audience_count"]
GROUPS = ["Both ≥60%", "Both <60%", "Audience ≥60% only", "Critics ≥60% only"]


def load_movies(path=DATA_PATH):
    raw = pd.read_csv(path)
    movies = raw.copy()
    for col in SCORES + COUNTS + ["runtime"]:
        movies[col] = pd.to_numeric(movies[col], errors="coerce")
    movies["release_year"] = pd.to_datetime(
        movies["original_release_date"], errors="coerce").dt.year.astype("Int64")
    movies["genre_list"] = (movies["genres"] .fillna("") .apply(lambda value: value.split(", ") if value else []))
    movies["gap_pp"] = movies["audience_rating"] - movies["tomatometer_rating"]
    movies["absolute_gap_pp"] = movies["gap_pp"].abs()
    paired = movies[SCORES].notna().all(axis=1)
    movies["verdict"] = pd.Series(pd.NA, index=movies.index, dtype="string")
    critic_positive = movies["tomatometer_rating"].ge(60)
    audience_positive = movies["audience_rating"].ge(60)
    movies.loc[paired, "verdict"] = np.select(
        [
            critic_positive[paired] & audience_positive[paired],
            ~critic_positive[paired] & ~audience_positive[paired],
            ~critic_positive[paired] & audience_positive[paired],
        ],
        GROUPS[:3],
        default=GROUPS[3],
    )
    return raw, movies


def filter_movies(movies, min_critics=20, min_audience=100, genres=()):
    valid = (movies[SCORES].notna().all(axis=1)
        & movies[SCORES].ge(0).all(axis=1)
        & movies[SCORES].le(100).all(axis=1))

    valid &= movies[COUNTS].notna().all(axis=1)
    valid &= movies["tomatometer_count"].ge(min_critics) & movies["audience_count"].ge(min_audience)
    if genres:
        valid &= movies["genre_list"].apply(
            lambda labels: bool(set(labels).intersection(genres)))
    return movies.loc[valid].copy()


def summary(movies):
    opposite = movies["verdict"].isin(GROUPS[2:])
    return pd.Series(
        {
            "titles": len(movies),
            "opposite_sides_pct": opposite.mean() * 100,
            "mean_absolute_gap_pp": movies["absolute_gap_pp"].mean(),
            "mean_gap_pp": movies["gap_pp"].mean(),
            "median_gap_pp": movies["gap_pp"].median(),
            "gap_at_least_20_pct": movies["absolute_gap_pp"].ge(20).mean() * 100,
            "pearson_r": movies[SCORES].corr().iloc[0, 1],
        }
    )


def genre_summary(movies, min_titles=100, selected_genres=()):
    expanded = movies.explode("genre_list").dropna(subset=["genre_list"]).copy()
    if selected_genres:
        expanded = expanded[expanded["genre_list"].isin(selected_genres)]
    expanded["opposite"] = expanded["verdict"].isin(GROUPS[2:])
    table = (
        expanded.groupby("genre_list")
        .agg(
            titles=("rotten_tomatoes_link", "nunique"),
            mean_gap_pp=("gap_pp", "mean"),
            median_gap_pp=("gap_pp", "median"),
            mean_absolute_gap_pp=("absolute_gap_pp", "mean"),
            opposite_sides_pct=("opposite", "mean"),
            )
            .rename_axis("genre")
            .reset_index()
        )
    table["opposite_sides_pct"] *= 100
    return table[table["titles"].ge(min_titles)].sort_values("mean_gap_pp")


def score_chart(movies):
    # Count titles in score bins so dense areas remain visible.
    bins = (
        alt.Chart(movies[SCORES])
        .mark_rect()
        .encode(
            x=alt.X(
                "tomatometer_rating:Q",
                bin=alt.Bin(step=5, extent=[0, 100]),
                scale=alt.Scale(domain=[0, 100]),
                title="Critic approval (%)",
            ),
            y=alt.Y(
                "audience_rating:Q",
                bin=alt.Bin(step=5, extent=[0, 100]),
                scale=alt.Scale(domain=[0, 100]),
                title="Audience approval (%)",
            ),
            color=alt.Color(
                "count():Q", scale=alt.Scale(scheme="orangered"), title="Titles"
            ),
            tooltip=[
                alt.Tooltip(
                    "tomatometer_rating:Q",
                    bin=alt.Bin(step=5, extent=[0, 100]),
                    title="Critic approval",
                ),
                alt.Tooltip(
                    "audience_rating:Q",
                    bin=alt.Bin(step=5, extent=[0, 100]),
                    title="Audience approval",
                ),
                alt.Tooltip("count():Q", title="Titles"),
            ],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"score": [0, 100]}))
        .mark_line(color="#28221F", strokeDash=[5, 4], strokeWidth=1)
        .encode(x="score:Q", y="score:Q")
    )
    return (bins + diagonal).properties(height=360).configure_view(stroke=None)


def genre_chart(table):
    bars = (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X(
                "mean_gap_pp:Q",
                title="Mean audience − critic approval (percentage points)",
                scale=alt.Scale(
                    domain=[
                        min(-10, table.mean_gap_pp.min() - 1),
                        max(10, table.mean_gap_pp.max() + 1),
                    ]
                ),
            ),
            y=alt.Y(
                "genre:N",
                axis=alt.Axis(labelLimit=230),
                sort=alt.SortField("mean_gap_pp", order="descending"),
                title=None,
            ),
            color=alt.condition(
                "datum.mean_gap_pp >= 0", alt.value("#b42a39"), alt.value("#27546b")
            ),
            tooltip=[
                alt.Tooltip("genre:N"),
                alt.Tooltip("titles:Q", format=","),
                alt.Tooltip("mean_gap_pp:Q", format=".2f"),
                alt.Tooltip("median_gap_pp:Q", format=".2f"),
            ],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#302b27")
        .encode(x="zero:Q")
    )
    return (
        (bars + zero)
        .properties(height=max(180, len(table) * 25))
        .configure_view(stroke=None)
    )


def export_table(movies):
    return movies[
        [
            "movie_title",
            "rotten_tomatoes_link",
            "release_year",
            "genres",
            *SCORES,
            *COUNTS,
            "gap_pp",
            "absolute_gap_pp",
            "verdict",
        ]
    ].sort_values("absolute_gap_pp", ascending=False)
