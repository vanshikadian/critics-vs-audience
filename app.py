import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from analysis import (
    GROUPS,
    PERIODS,
    PROCESSED_DIR,
    PROFILES,
    compare_profiles,
    comparison_chart,
    export_table,
    filter_movies,
    genre_chart,
    genre_summary,
    load_prepared,
    period_chart,
    period_genres,
    period_summary,
    score_chart,
    standardized_genres,
    summary,
)

ROOT = Path(__file__).parent
SOURCE_URL = "https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"

st.set_page_config(
    page_title="Split Decision · Critics & audiences", page_icon="🎬", layout="wide"
)
st.markdown(
    f'<style>{(ROOT / "assets/styles.css").read_text()}</style>', unsafe_allow_html=True
)
alt.data_transformers.disable_max_rows()


@st.cache_data
def read_data(csv_stamp, metadata_stamp):
    return load_prepared()


def show_table(table, names=None):
    st.dataframe(
        table.round(2).rename(columns=names or {}), hide_index=True, width="stretch"
    )


def download(label, table, filename):
    st.download_button(
        label, table.to_csv(index=False).encode("utf-8"), filename, "text/csv"
    )


def metrics(movies):
    values = summary(movies)
    columns = st.columns(3)
    columns[0].metric("Titles in this view", f"{len(movies):,}")
    columns[1].metric("Opposite sides of 60%", f"{values.opposite_sides_pct:.1f}%")
    columns[2].metric("Mean absolute gap", f"{values.mean_absolute_gap_pp:.1f} pp")
    return values


st.html(
    """<header class="movie-header">
<p class="kicker">CRITICS & AUDIENCES / OCTOBER 2020 SNAPSHOT</p>
<h1 class="movie-mark">Split <b>Decision</b></h1>
<div class="billing"><p>Same title.<br><em>Different verdict?</em></p></div>
</header>"""
)
st.write(
    "For movie fans curious about the scores: similar overall averages can hide very different verdicts on "
    "individual titles. Explore the gaps, then test whether the genre patterns hold up.")

try:
    metadata, movies = read_data(
        (PROCESSED_DIR / "movie_scores.csv").stat().st_mtime_ns,
        (PROCESSED_DIR / "preparation.json").stat().st_mtime_ns,
    )

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(
        "The prepared data are missing or inconsistent. Follow the README to obtain the raw CSV, then run `python scripts/prepare_data.py` and refresh."
    )
    st.markdown(f"[Dataset source]({SOURCE_URL})")
    with st.expander("Details"):
        st.code(str(error))
    st.stop()

with st.sidebar:
    st.header("Set the screening")
    min_critics = st.slider("Minimum critic reviews", 5, 600, 20, 5)
    min_audience = st.select_slider(
        "Minimum audience ratings",
        options=[0, 50, 100, 500, 1000, 10000, 100000],
        value=100,
    )
    choices = sorted({genre for labels in movies.genre_list for genre in labels})
    selected_genres = st.multiselect("Genres (any selected)", choices)
    min_titles = st.slider("Minimum titles per displayed genre", 10, 500, 100, 10)
    st.caption(
        "Leave genres blank for all titles. Counts filter the sample; they do not guarantee reliable or representative scores."
    )
    st.caption(
        "These controls set the Overview, Genres and Release periods views. The Filter test uses its own count presets and keeps this genre selection."
    )
    st.markdown(f"[Dataset source]({SOURCE_URL})")

filtered = filter_movies(movies, min_critics, min_audience, selected_genres)
st.caption(
    f'{metadata["rows"]:,} source titles · {len(filtered):,} in the current exploration view · Each title counts once before genre grouping.'
)

st.caption(
    "Gap = audience - critic approval, in percentage points (pp). Positive: audience higher. Negative: critics higher. Scores are approval percentages, not average stars."
)

overview, by_genre, sensitivity, by_period, methods = st.tabs(
    ["Overview", "Genres", "Filter test", "Release periods", "Data & methods"]
)

with overview:
    st.subheader("How much agreement does the average hide?")
    if filtered.empty:
        st.info(
            "No titles match these filters. Lower the review minimums or clear the genre selection."
        )

    else:
        values = metrics(filtered)
        st.caption(
            "“Opposite sides” means one score is at least 60% and the other is below 60%. It does not measure the size of the gap.")
        st.write(
            "Darker squares contain more titles. Above the dashed equal-score line, audience approval is higher; below it, critic approval is higher.")

        st.altair_chart(score_chart(filtered), width="stretch")
        counts = (
            filtered.verdict.value_counts()
            .reindex(GROUPS, fill_value=0)
            .rename_axis("Verdict")
            .reset_index(name="Titles"))

        counts["Share (%)"] = counts.Titles / len(filtered) * 100
        show_table(counts)
        correlation = (
            f"{values.pearson_r:.2f}"
            if math.isfinite(values.pearson_r)
            else "undefined for this view")

        st.write(
            f"The signed mean gap is {values.mean_gap_pp:+.2f} pp, while the mean absolute gap is {values.mean_absolute_gap_pp:.2f} pp. Differences in opposite directions can cancel in the signed average. Here, {values.gap_at_least_20_pct:.2f}% of titles are at least 20 points apart.")

        st.caption(
            f"Median signed gap: {values.median_gap_pp:+.1f} pp. Pearson r: {correlation}. Correlation describes association, not agreement. The 20-point cutoff is exploratory.")
        with st.expander("See the distribution of individual gaps"):
            histogram = (
                alt.Chart(filtered[["gap_pp"]])
                .mark_bar(color="#AC292E")
                .encode(
                    x=alt.X(
                        "gap_pp:Q",
                        bin=alt.Bin(step=5, extent=[-100, 100]),
                        title="Audience − critic approval (pp)",
                        scale=alt.Scale(domain=[-100, 100]),
                    ),
                    y=alt.Y("count():Q", title="Titles"),
                    tooltip=[
                        alt.Tooltip(
                            "gap_pp:Q",
                            bin=alt.Bin(step=5, extent=[-100, 100]),
                            title="Gap range",
                        ),
                        alt.Tooltip("count():Q", title="Titles"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(histogram, width="stretch")

with by_genre:
    st.subheader("Which genres lean in which direction?")
    st.write(
        "Each title has equal weight. Red bars mean higher audience approval; blue bars mean higher critic approval. Compare the median as a check on the mean.")

    statistic = st.radio(
        "Compare genres using", ["Mean gap", "Median gap"], horizontal=True)

    genres = genre_summary(filtered, min_titles, selected_genres)
    if genres.empty:
        st.info(
            "No genre has enough titles in this view. Lower the genre minimum or broaden the filters.")

    else:
        chart = genre_chart( genres, "mean_gap_pp" if statistic == "Mean gap" else "median_gap_pp")

        st.altair_chart(chart, width="stretch")
        show_table(
            genres,
            {
                "genre": "Genre",
                "titles": "Titles",
                "mean_gap_pp": "Mean gap (pp)",
                "median_gap_pp": "Median gap (pp)",
                "mean_absolute_gap_pp": "Mean absolute gap (pp)",
                "opposite_sides_pct": "Opposite sides (%)",
            },
        )
    st.caption(
        "Genre groups overlap: one title can appear in several groups. A larger signed gap does not necessarily mean more opposite-side verdicts. The Filter test checks how much these averages depend on the included titles.")

with sensitivity:
    st.subheader("Does the finding survive a different filter?")
    st.write(
        "Compare two review-count rules on the same historical source. The stricter rule changes which titles are included; this is a sensitivity check, not evidence that anyone changed their opinion.")

    left_col, right_col = st.columns(2)
    labels = list(PROFILES)

    profile_a = left_col.selectbox(
        "Filter A · critic / audience minimum", labels, index=1)


    profile_b = right_col.selectbox(
        "Filter B · critic / audience minimum", labels, index=3)

    st.caption(
        "Red circles show A; teal diamonds show B. Only the genre selection and minimum genre size from the sidebar apply here. The two review-count presets replace the sidebar count filters.")

    left = filter_movies(movies, *PROFILES[profile_a], selected_genres)
    right = filter_movies(movies, *PROFILES[profile_b], selected_genres)
    for column, label, data in [(left_col, "A", left), (right_col, "B", right)]:
        values = summary(data)
        column.write(f"**{label}: {len(data):,} titles**")
        column.caption(
            f"Mean gap: {values.mean_gap_pp:+.2f} pp; mean absolute gap: {values.mean_absolute_gap_pp:.2f} pp."
            if len(data)
            else "No titles meet this rule.")

    if profile_a == profile_b:
        st.info(
            "The same rule is selected twice, so the results are identical. Choose another rule to compare.")

    compared = compare_profiles(
        movies, PROFILES[profile_a], PROFILES[profile_b], min_titles, selected_genres)

    supported = compared[compared.enough_titles]
    if supported.empty:
        st.info(
            "No genre meets the minimum title count in both views. Lower the genre minimum or broaden the genre selection.")

    else:
        st.altair_chart(comparison_chart(supported, "A", "B"), width="stretch")
        flips = supported[supported.direction.eq("Direction changes")]
        if len(flips):
            st.write(
                "Direction changes beyond ±1 pp: **"
                + ", ".join(flips.genre)
                + "**. Check the mean, median and sample sizes before drawing a conclusion.")
        else:
            st.write(
                "No qualifying genre changes from below −1 pp to above +1 pp, or vice versa, in this comparison.")

    show_table(
        compared,
        {
            "genre": "Genre",
            "titles_A": "Titles A",
            "titles_B": "Titles B",
            "mean_gap_pp_A": "Mean A (pp)",
            "mean_gap_pp_B": "Mean B (pp)",
            "median_gap_pp_A": "Median A (pp)",
            "median_gap_pp_B": "Median B (pp)",
            "enough_titles": "Meets both minimums",
            "change_pp": "B − A (pp)",
            "direction": "Comparison",
        },
    )
    st.caption(
        "“Near zero” means within ±1 pp in at least one view—an exploratory tolerance, not a significance test. Keeping the same direction does not establish a robust causal effect. Both views reuse many of the same titles.")
    download(
        "Download filter comparison (CSV)",
        compared,
        "split_decision_filter_comparison.csv",
    )

with by_period:
    st.subheader("Is the pattern similar across release periods?")
    st.write(
        "Group titles by their original release year. These are release cohorts, not the dates the reviews were written; all scores come from the same 2020 snapshot.")

    unknown = int(filtered.release_year.isna().sum())

    st.caption(
        f"{unknown:,} titles in the current view have no release year and are excluded only from this section. The last period ends with a partial 2020; the periods span different numbers of years.")


    periods = period_summary(filtered)
    if periods.empty:
        st.info("No titles with known release years match these filters.")
    else:
        period_plot = (
            alt.Chart(periods)
            .mark_bar(color="#AC292E")
            .encode(
                x=alt.X(
                    "period:N", sort=PERIODS, title=None, axis=alt.Axis(labelAngle=0)
                ),
                y=alt.Y("mean_gap_pp:Q", title="Mean audience - critic approval (pp)"),
                tooltip=[
                    "period:N",
                    alt.Tooltip("titles:Q", format=","),
                    alt.Tooltip("mean_gap_pp:Q", format=".2f"),
                ],
            ).properties(height=240)
        )
        st.altair_chart(period_plot, width="stretch")
        show_table(
            periods,
            {
                "period": "Release period",
                "titles": "Titles",
                "mean_gap_pp": "Mean gap (pp)",
                "median_gap_pp": "Median gap (pp)",
                "mean_absolute_gap_pp": "Mean absolute gap (pp)",
                "opposite_sides_pct": "Opposite sides (%)",
                "gap_at_least_20_pct": "Gap ≥20 pp (%)",
                "pearson_r": "Pearson r",
            },
        )
        minimum_cell = st.slider("Minimum titles per genre and release period", 10, 100, 20, 10)

        cells = period_genres(filtered, selected_genres)
        if cells[cells.titles.ge(minimum_cell)].empty:
            st.info("No genre-period group meets this minimum.")
        else:
            st.altair_chart(period_chart(cells, minimum_cell), width="stretch")
        st.caption(
            "Blank cells have no titles or fewer than the selected minimum; they do not mean a zero gap. Red: audience higher. Blue: critics higher. Overlapping genres and selection into this catalog limit comparisons.")

        with st.expander("What if every genre had the same release-period mix?"):
            standardized, weights = standardized_genres(
                filtered, minimum_cell, selected_genres)

            st.write(
                "Recalculate each genre's mean using the same period weights: the shares of all known-year titles in this filtered view. This isolates the arithmetic effect of different period mixes; it does not control for all confounding or establish a genre effect.")

            show_table(
                weights.rename("Share")
                .rename_axis("Release period")
                .reset_index()
                .assign(Share=lambda d: d.Share * 100),
                {"Share": "Reference share (%)"},
            )
            if standardized.empty:
                st.info(
                    "No genre has enough titles in every represented release period. Lower the cell minimum or broaden the filters.")

            else:
                show_table(
                    standardized,
                    {
                        "genre": "Genre",
                        "titles": "Known-year titles",
                        "observed_gap_pp": "Observed mean (pp)",
                        "same_period_mix_gap_pp": "Same-period-mix mean (pp)",
                        "change_pp": "Change (pp)",
                        "periods": "Periods",
                    },
                )
            st.caption(
                "Only genres meeting the cell minimum in every represented period are included. The observed comparison here also excludes unknown-year titles, so it may differ from the Genres tab.")
            download(
                "Download period-mix comparison (CSV)",
                standardized,
                "split_decision_period_mix.csv",
            )
        download(
            "Download all genre-period groups (CSV)",
            cells,
            "split_decision_release_periods.csv",
        )

with methods:
    st.subheader("What went into these comparisons?")
    st.markdown(
        f"""**Source:** [Stefano Leone's Rotten Tomatoes dataset]({SOURCE_URL}), collected through **October 31, 2020**. There are **17,712 source titles and 22 original columns**, including some television and other non-theatrical titles. The publisher lists CC0. This is a historical catalog snapshot, not a random sample or today's scores.

**Preparation:** the saved table keeps every source title and adds numeric fields, release periods, score gaps, verdict groups and validity flags. Source IDs distinguish repeated names. The raw CSV is unchanged. No scores, counts or release years are imputed.

**Inclusion:** both scores must be between 0 and 100, and both counts must meet the selected minimums. Zero scores are valid. Missing genres stay in the overall comparison but cannot enter a genre group; missing release years are excluded only from release-period analysis. Every title has equal weight; review counts are filters, not weights.

**No unnecessary model:** this is a descriptive comparison, not a prediction task. No scaling, categorical encoding or machine learning is needed for the stated questions.""")

    quality = metadata["quality"]
    show_table(
        pd.DataFrame(
            {
                "Check": [
                    "Exact duplicate rows",
                    "Repeated source IDs",
                    "Repeated title names",
                    "Paired valid scores",
                    "Scores and counts available",
                    "Fresh + rotten counts differ from critic total",
                ],
                "Titles": [
                    quality["exact_duplicates"],
                    quality["duplicate_ids"],
                    quality["repeated_title_strings"],
                    metadata["paired_scores"],
                    metadata["comparison_eligible"],
                    quality["critic_count_mismatches"],
                ],
            }
        )
    )
    with st.expander("Missing values in the original table"):
        show_table(
            pd.Series(quality["missing"], name="Missing titles")
            .rename_axis("Original field")
            .reset_index())

    st.markdown(
        """**Important limits:** critic and audience scores are approval percentages from different populations and scoring rules—not average stars. Audience verification status is absent. Counts are highly skewed and do not show representativeness. The publisher reports different scraping times for movie and review data; the 279 count mismatches are why we use published approval scores directly.

These analyses cannot explain individual opinions, actual enjoyment, review bombing, a causal genre effect, or changes in taste over time. A 60% threshold crossing and a large numerical gap measure different things. The support cutoffs, genre minimums and ±1 pp tolerance are analytical choices, not confidence levels.

**Reproduce:** `python scripts/prepare_data.py` rebuilds `data/processed/movie_scores.csv` and its quality/provenance record from the documented raw CSV. `python scripts/check_project.py` checks the pipeline and analysis; `python scripts/check_app.py` exercises the app. The analysis notebook contains saved graphs and methods. Nothing here needs an API key or live data service.

[Rotten Tomatoes score definitions](https://www.rottentomatoes.com/about)""")

with st.expander("Look at the titles / download this view"):
    st.caption(
        "This table uses the sidebar filters, not the two Filter test presets. It is sorted by absolute gap. Repeated names remain separate source IDs.")

    query = st.text_input(
        "Find a title", placeholder="Search within the current filters")

    exported = export_table(filtered)
    visible = (
        exported[exported.movie_title.str.contains(query, case=False, regex=False)]
        if query
        else exported)

    st.caption(f"{len(visible):,} matching titles.")
    show_table(visible)
    download("Download filtered titles (CSV)", visible, "split_decision_titles.csv")
