import math
from pathlib import Path

import altair as alt
import streamlit as st

from analysis import (
    GROUPS,
    export_table,
    filter_movies,
    genre_chart,
    genre_summary,
    load_movies,
    score_chart,
    summary,
)

st.set_page_config(
    page_title="Split Decision · Critics & audiences", page_icon="🎬", layout="wide"
)
st.markdown(
    f'<style>{(Path(__file__).parent / "assets/styles.css").read_text()}</style>',
    unsafe_allow_html=True,
)
alt.data_transformers.disable_max_rows()


@st.cache_data
def read_data():
    return load_movies()


st.html(
    """<header class="movie-header">
<p class="kicker">A MOVIE DATA PROJECT / OCTOBER 2020 SNAPSHOT</p>
<h1 class="movie-mark">Split <b>Decision</b></h1>
<div class="billing"><p>Same title.<br><em>Different verdict?</em></p></div>
</header>"""
)
st.write(
    "For movie fans curious about the scores: how often do critics and audiences disagree—and does the gap vary across genres?"
)
try:
    raw, movies = read_data()
except FileNotFoundError:
    st.error(
        "The movie table is missing. Download rotten_tomatoes_movies.csv from the dataset linked in the sidebar and place it in data/raw/, then refresh."
    )
    st.stop()

with st.sidebar:
    st.header("Set the screening")
    min_critics = st.slider("Minimum critic reviews", 5, 600, 20, 5)
    min_audience = st.select_slider(
        "Minimum audience ratings",
        options=[0, 100, 500, 1000, 10000, 100000],
        value=100,
    )
    choices = sorted({genre for labels in movies["genre_list"] for genre in labels})
    selected_genres = st.multiselect("Genres (any selected)", choices)
    st.caption(
        "These are analyst-selected support filters, not guarantees of reliable or representative scores. Leave genres blank to include all titles."
    )
    st.markdown(
        "[Dataset source](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)"
    )

filtered = filter_movies(movies, min_critics, min_audience, selected_genres)
st.caption(
    f"{len(raw):,} source titles → {len(filtered):,} with both scores and counts meeting these filters. Each title counts once."
)
if filtered.empty:
    st.info(
        "No titles match these filters. Lower the review minimums or clear the genre selection."
    )
    st.stop()

stats = summary(filtered)
cols = st.columns(3)
cols[0].metric("Titles in this view", f"{len(filtered):,}")
cols[1].metric("Opposite sides of 60%", f"{stats.opposite_sides_pct:.1f}%")
cols[2].metric("Mean absolute gap", f"{stats.mean_absolute_gap_pp:.1f} pp")
st.caption(
    "“Opposite sides” means one score is at least 60% and the other is below 60%. A percentage point (pp) is the difference between two percentages."
)

gap_tab, genre_tab = st.tabs(["01 / The score gap", "02 / By genre"])
with gap_tab:
    st.subheader("How often do they disagree?")
    st.write(
        "Darker squares contain more titles. Above the equal-score diagonal, audience approval is higher; below it, critic approval is higher."
    )
    st.altair_chart(score_chart(filtered), width="stretch")
    counts = (
        filtered["verdict"]
        .value_counts()
        .reindex(GROUPS, fill_value=0)
        .rename_axis("Verdict")
        .reset_index(name="Titles")
    )
    counts["Share (%)"] = (counts["Titles"] / len(filtered) * 100).round(1)
    st.dataframe(counts, hide_index=True, width="stretch")
    correlation = (
        f"{stats.pearson_r:.2f}"
        if math.isfinite(stats.pearson_r)
        else "undefined for this view"
    )
    st.caption(
        f"{stats.gap_at_least_20_pct:.1f}% have an absolute gap of at least 20 pp (an exploratory cutoff). Signed mean: {stats.mean_gap_pp:+.2f} pp; median: {stats.median_gap_pp:+.1f} pp; Pearson r: {correlation}. Correlation can be high even when individual titles have large gaps."
    )
with genre_tab:
    st.subheader("Which genres lean in which direction?")
    st.write(
        "Red bars lean toward audience approval; blue bars lean toward critic approval. These are averages of title-level gaps, with equal weight per title."
    )
    min_titles = st.slider("Minimum titles per displayed genre", 10, 500, 100, 10)
    genres = genre_summary(filtered, min_titles, selected_genres)
    if genres.empty:
        st.info(
            "No genre has enough titles in this view. Lower the genre minimum or broaden the filters."
        )
    else:
        st.altair_chart(genre_chart(genres), width="stretch")
        st.dataframe(
            genres.round(2).rename(
                columns={
                    "genre": "Genre",
                    "titles": "Titles",
                    "mean_gap_pp": "Mean gap (pp)",
                    "median_gap_pp": "Median gap (pp)",
                    "mean_absolute_gap_pp": "Mean absolute gap (pp)",
                    "opposite_sides_pct": "Opposite sides (%)",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "A title can belong to several genres, so genre totals overlap. Genre differences are descriptive, not explanations of why people rated a title. Some patterns change with the support filters; checking that sensitivity is part of the next stage."
    )

with st.expander("Look at the titles / download this view"):
    query = st.text_input(
        "Find a title", placeholder="Search within the current filters"
    )
    exported = export_table(filtered)
    visible = (
        exported[exported["movie_title"].str.contains(query, case=False, regex=False)]
        if query
        else exported
    )
    st.caption(
        f"{len(visible):,} matching titles. Sorted by absolute gap; repeated names are kept as separate source IDs."
    )
    st.dataframe(visible, hide_index=True, width="stretch")
    st.download_button(
        "Download filtered titles (CSV)",
        visible.to_csv(index=False).encode("utf-8"),
        "split_decision_titles.csv",
        "text/csv",
    )

with st.expander("What these data can—and cannot—tell us"):
    st.markdown(
        """The source is Stefano Leone’s **October 31, 2020** Rotten Tomatoes snapshot: **17,712 titles and 22 columns**. It includes some television and other non-theatrical titles. This is a historical sample, not the current catalog.

Critic and audience scores are **approval percentages**, from different groups and scoring rules—not average star ratings. The table does not identify audience verification status or explain individual opinions. A gap does not establish quality, enjoyment, review bombing, or a cause.

Titles missing either score or either count are excluded from these comparisons. Zero scores remain valid. Missing genres are retained in the overall comparison but cannot appear in a genre bar. We do not impute scores or weight title averages by review count.

The publisher notes that review and movie data were scraped at different times. Fresh + rotten critic counts differ from the reported critic total for 279 titles, so we use the published approval score directly. Audience counts are highly skewed and are not evidence of a representative sample.

[Source and publisher notes](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) · [Rotten Tomatoes score definitions](https://www.rottentomatoes.com/about)"""
    )
