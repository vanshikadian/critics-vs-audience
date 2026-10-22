"""Movie-level comparisons for the October 2020 Rotten Tomatoes snapshot."""

import hashlib
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/rotten_tomatoes_movies.csv"
SCORES = ["tomatometer_rating", "audience_rating"]
COUNTS = ["tomatometer_count", "audience_count"]
PIPELINE_VERSION = 1
PROCESSED_DIR = ROOT / "data/processed"
PERIODS = ["Before 1980", "1980–1999", "2000–2009", "2010–2020"]
PROFILES = {
    "Broad · 10 / 50": (10, 50),
    "Default · 20 / 100": (20, 100),
    "Stricter · 50 / 500": (50, 500),
    "Strict · 50 / 1,000": (50, 1000),
    "Very strict · 100 / 1,000": (100, 1000),
}
GROUPS = ["Both ≥60%", "Both <60%", "Audience ≥60% only", "Critics ≥60% only"]


def load_movies(path=DATA_PATH):
    raw = pd.read_csv(path)
    return raw, prepare_movies(raw)


def prepare_movies(raw):
    movies = raw.copy()
    for col in SCORES + COUNTS + ["runtime"]:
        movies[col] = pd.to_numeric(movies[col], errors="coerce")

    movies["release_year"] = pd.to_datetime(
        movies["original_release_date"], errors="coerce").dt.year.astype("Int64")

    movies["genre_list"] = (
        movies["genres"]
        .fillna("")
        .apply(lambda value: value.split(", ") if value else []))

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

    movies["release_period"] = pd.cut(
        movies["release_year"].astype(float),
        [-float("inf"), 1979, 1999, 2009, 2020],
        labels=PERIODS,
    ).astype("string")

    movies["scores_valid"] = (
        movies[SCORES].notna().all(axis=1)
        & movies[SCORES].ge(0).all(axis=1)
        & movies[SCORES].le(100).all(axis=1))

    movies["counts_valid"] = movies[COUNTS].notna().all(axis=1) & movies[COUNTS].gt(0).all(axis=1)
    movies["comparison_eligible"] = movies["scores_valid"] & movies["counts_valid"]
    return movies


def filter_movies(movies, min_critics=20, min_audience=100, genres=()):
    valid = (
        movies[SCORES].notna().all(axis=1)
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
            "pearson_r": (
                movies[SCORES].corr().iloc[0, 1]
                if len(movies) > 1 and movies[SCORES].nunique().min() > 1
                else float("nan")
            ),
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
        .encode(x="score:Q", y="score:Q"))

    return (bins + diagonal).properties(height=360).configure_view(stroke=None)


def genre_chart(table, statistic="mean_gap_pp"):
    label = "Median" if statistic == "median_gap_pp" else "Mean"
    bars = (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{statistic}:Q",
                title=f"{label} audience - critic approval (percentage points)",
                scale=alt.Scale(
                    domain=[
                        min(-10, table[statistic].min() - 1),
                        max(10, table[statistic].max() + 1),
                    ]
                ),
            ),
            y=alt.Y(
                "genre:N",
                axis=alt.Axis(labelLimit=230),
                sort=alt.SortField(statistic, order="descending"),
                title=None,
            ),
            color=alt.condition(
                f"datum.{statistic} >= 0", alt.value("#b42a39"), alt.value("#27546b")
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
        ]].sort_values("absolute_gap_pp", ascending=False)


def load_prepared(directory=PROCESSED_DIR):
    path = directory / "movie_scores.csv"
    metadata = json.loads((directory / "preparation.json").read_text())
    source = json.loads((ROOT / "data/source.json").read_text())
    if (
        metadata.get("pipeline_version") != PIPELINE_VERSION
        or metadata["raw_sha256"] != source["sha256"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["processed_sha256"]
    ):
        raise ValueError(
            "Prepared files do not match their recorded dataset version. Rebuild them with scripts/prepare_data.py.")

    movies = pd.read_csv(
        path,
        dtype={
            "release_year": "Int64",
            "verdict": "string",
            "release_period": "string",
        },
    )
    if len(movies) != metadata["rows"] or not movies.rotten_tomatoes_link.is_unique:
        raise ValueError(
            "Prepared title IDs or row count are inconsistent. Rebuild the prepared files.")

    movies["genre_list"] = movies.genres.fillna("").apply(lambda value: value.split(", ") if value else [])
    return metadata, movies


def compare_profiles(movies, left, right, min_titles=100, genres=()):
    tables = []
    for label, thresholds in [("A", left), ("B", right)]:
        selected = filter_movies(movies, *thresholds, genres)
        table = genre_summary(selected, 0, genres)[
            ["genre", "titles", "mean_gap_pp", "median_gap_pp"]
        ]
        tables.append(
            table.rename(
                columns={ col: f"{col}_{label}" for col in table.columns if col != "genre"}
            )
        )

    compared = tables[0].merge(tables[1], on="genre", how="outer")
    compared[["titles_A", "titles_B"]] = (
        compared[["titles_A", "titles_B"]].fillna(0).astype(int))

    compared["enough_titles"] = compared.titles_A.ge(min_titles) & compared.titles_B.ge(min_titles)
    compared["change_pp"] = compared.mean_gap_pp_B - compared.mean_gap_pp_A

    def direction(row):
        if not row.enough_titles:
            return "Too few titles"
        if abs(row.mean_gap_pp_A) <= 1 or abs(row.mean_gap_pp_B) <= 1:
            return "Near zero in at least one view"
        if row.mean_gap_pp_A * row.mean_gap_pp_B < 0:
            return "Direction changes"
        return (
            "Audience higher in both"
            if row.mean_gap_pp_A > 0
            else "Critics higher in both"
        )

    compared["direction"] = (compared.apply(direction, axis=1) if len(compared) else pd.Series(dtype="str"))

    return compared.sort_values("genre").reset_index(drop=True)


def comparison_chart(table, label_a="A", label_b="B"):
    data = table[table.enough_titles].copy()
    bound = max(10, abs(data[["mean_gap_pp_A", "mean_gap_pp_B"]]).max().max() + 1)
    line = (
        alt.Chart(data)
        .mark_rule(color="#a59b85", strokeWidth=2)
        .encode(
            x=alt.X(
                "mean_gap_pp_A:Q",
                scale=alt.Scale(domain=[-bound, bound]),
                title="Mean audience − critic approval (pp)",
            ),
            x2="mean_gap_pp_B:Q",
            y=alt.Y("genre:N", title=None, axis=alt.Axis(labelLimit=230)),
        )
    )
    points = data.melt(
        id_vars=["genre"],
        value_vars=["mean_gap_pp_A", "mean_gap_pp_B"],
        var_name="view",
        value_name="gap",
    )

    points["view"] = points["view"].map(
        {"mean_gap_pp_A": label_a, "mean_gap_pp_B": label_b})

    dots = (
        alt.Chart(points)
        .mark_point(filled=True, size=85)
        .encode(
            x="gap:Q",
            y=alt.Y("genre:N", title=None),
            color=alt.Color(
                "view:N",
                scale=alt.Scale(
                    domain=[label_a, label_b], range=["#AC292E", "#17656B"]
                ),
                title="Review-count filter",
            ),
            shape=alt.Shape(
                "view:N",
                scale=alt.Scale(domain=[label_a, label_b], range=["circle", "diamond"]),
                legend=alt.Legend(title="Review-count filter"),
            ),
            tooltip=[
                "genre:N",
                "view:N",
                alt.Tooltip("gap:Q", format=".2f", title="Mean gap (pp)"),
            ],
        )
    )

    zero = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#28221F", strokeDash=[4, 4])
        .encode(x="zero:Q")
    )

    return (
        (line + dots + zero)
        .properties(height=max(180, len(data) * 26))
        .configure_view(stroke=None)
    )


def period_summary(movies, genres=()):
    known = movies.dropna(subset=["release_period"])
    if genres:
        known = known[
            known.genre_list.apply(
                lambda labels: bool(set(labels).intersection(genres))
            )
        ]

    rows = []
    for period in PERIODS:
        group = known[known.release_period.eq(period)]
        if len(group):
            rows.append({"period": period, **summary(group).to_dict()})
    return pd.DataFrame(
        rows,
        columns=[
            "period",
            "titles",
            "opposite_sides_pct",
            "mean_absolute_gap_pp",
            "mean_gap_pp",
            "median_gap_pp",
            "gap_at_least_20_pct",
            "pearson_r",
        ],
    )


def period_genres(movies, genres=()):
    tables = []
    for period in PERIODS:
        subset = movies[movies.release_period.eq(period).fillna(False)]
        table = genre_summary(subset, 0, genres)
        table["period"] = period
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def period_chart(table, min_titles=20):
    data = table[table.titles.ge(min_titles)].copy()
    bound = max(5, abs(data.mean_gap_pp).max())
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("period:N", sort=PERIODS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("genre:N", title=None, axis=alt.Axis(labelLimit=230)),
            color=alt.Color(
                "mean_gap_pp:Q",
                scale=alt.Scale(
                    domain=[-bound, 0, bound], range=["#17656B", "#fff7df", "#AC292E"]
                ),
                title="Mean gap (pp)",
            ),
            tooltip=[
                "genre:N",
                "period:N",
                alt.Tooltip("titles:Q", format=","),
                alt.Tooltip("mean_gap_pp:Q", format=".2f", title="Mean gap (pp)"),
            ],
        )
        .properties(height=max(180, data.genre.nunique() * 26))
        .configure_view(stroke=None)
    )


def standardized_genres(movies, min_per_period=20, genres=()):
    known = movies.dropna(subset=["release_period"])
    weights = known.release_period.value_counts(normalize=True).reindex(PERIODS, fill_value=0)

    periods = weights[weights.gt(0)].index.tolist()
    cells = period_genres(known, genres)
    rows = []
    for genre, group in cells.groupby("genre"):
        group = group.set_index("period").reindex(periods)
        if (
            len(periods) < 2
            or group.titles.isna().any()
            or not group.titles.ge(min_per_period).all()
        ):
            continue
        total = group.titles.sum()
        observed = (group.mean_gap_pp * group.titles).sum() / total
        standardized = (group.mean_gap_pp * weights.loc[periods]).sum()
        rows.append(
            {
                "genre": genre,
                "titles": int(total),
                "observed_gap_pp": observed,
                "same_period_mix_gap_pp": standardized,
                "change_pp": standardized - observed,
                "periods": len(periods),
            }
        )
    return (
        pd.DataFrame(
            rows,
            columns=[
                "genre",
                "titles",
                "observed_gap_pp",
                "same_period_mix_gap_pp",
                "change_pp",
                "periods",
            ],
        ).sort_values("same_period_mix_gap_pp"),weights,
    )
