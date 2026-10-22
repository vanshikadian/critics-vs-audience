"""Check the dataset, calculations, filters, and CSV export."""

import hashlib
import io
import json
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import (
    DATA_PATH,
    GROUPS,
    SCORES,
    export_table,
    filter_movies,
    genre_summary,
    load_movies,
    summary,
    compare_profiles,
    load_prepared,
    period_summary,
    period_genres,
    standardized_genres,
)


def audit(raw):
    return {
        "rows": len(raw),
        "columns": len(raw.columns),
        "exact_duplicates": int(raw.duplicated().sum()),
        "duplicate_ids": int(raw.rotten_tomatoes_link.duplicated().sum()),
        "repeated_title_strings": int(raw.movie_title.duplicated().sum()),
        "missing": raw.isna().sum().to_dict(),
        "scores_outside_0_100": {
            col: int((raw[col].notna() & ~raw[col].between(0, 100)).sum())
            for col in SCORES
        },
        "nonpositive_counts": {
            col: int(raw[col].le(0).sum())
            for col in ["tomatometer_count", "audience_count"]
        },
        "critic_count_mismatches": int(
            (
                raw.tomatometer_count.notna()
                & raw.tomatometer_fresh_critics_count.add(
                    raw.tomatometer_rotten_critics_count
                ).ne(raw.tomatometer_count)
            ).sum()
        ),
    }


def main():
    source = json.loads((ROOT / "data/source.json").read_text())
    assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == source["sha256"]
    raw, movies = load_movies()
    quality = audit(raw)
    assert raw.shape == (17712, 22)
    assert quality["duplicate_ids"] == quality["exact_duplicates"] == 0
    assert quality["repeated_title_strings"] == 606
    assert not any(quality["scores_outside_0_100"].values())
    assert not any(quality["nonpositive_counts"].values())
    assert quality["critic_count_mismatches"] == 279
    assert movies[SCORES].notna().all(axis=1).sum() == 17407
    base = filter_movies(movies)
    assert len(base) == 10503
    np.testing.assert_allclose(
        base.gap_pp, base.audience_rating - base.tomatometer_rating)

    assert base.rotten_tomatoes_link.is_unique
    assert base.verdict.value_counts().to_dict() == dict(
        zip(GROUPS, [5122, 3069, 1123, 1189]))

    stats = summary(base)
    assert np.isclose(stats.opposite_sides_pct, 2312 / 10503 * 100)
    assert filter_movies(movies, 600, 100).empty
    horror = filter_movies(movies, genres=["Horror"])
    assert horror.genre_list.apply(lambda labels: "Horror" in labels).all()
    assert len(horror) == 1062
    assert genre_summary(base).titles.sum() > len(base)
    assert genre_summary(base, min_titles=20000).empty
    assert base.genres.isna().sum() == 1
    assert (base[SCORES] == 0).any().any()
    export = export_table(base)
    assert len(pd.read_csv(io.StringIO(export.to_csv(index=False)))) == len(base)
    assert export.rotten_tomatoes_link.is_unique
    metadata, prepared = load_prepared()
    assert len(prepared) == len(raw) and prepared.rotten_tomatoes_link.is_unique
    assert prepared.comparison_eligible.sum() == 17406
    assert metadata["raw_sha256"] == source["sha256"]
    for col in [
        "gap_pp",
        "absolute_gap_pp",
        "tomatometer_rating",
        "audience_rating",
        "tomatometer_count",
        "audience_count",
    ]:
        np.testing.assert_allclose(prepared[col], movies[col], equal_nan=True)
    pd.testing.assert_series_equal(filter_movies(prepared).gap_pp, base.gap_pp)
    assert prepared.verdict.fillna("").tolist() == movies.verdict.fillna("").tolist()
    assert prepared.release_year.equals(movies.release_year)
    comparison = compare_profiles(prepared, (20, 100), (50, 1000))
    flipped = comparison.set_index("genre").loc["Horror"]
    assert flipped.direction == "Direction changes"
    assert flipped.titles_A == 1062 and flipped.titles_B == 553
    assert flipped.median_gap_pp_B == -1
    strict_horror = filter_movies(prepared, 50, 1000, ["Horror"])
    assert np.isclose(flipped.mean_gap_pp_B, strict_horror.gap_pp.mean())
    same = compare_profiles(prepared, (20, 100), (20, 100))
    np.testing.assert_allclose(same.change_pp, 0)
    assert not same.direction.eq("Direction changes").any()
    assert compare_profiles(prepared, (600, 100), (600, 100)).empty
    periods = period_summary(base)
    assert periods.titles.astype(int).tolist() == [932, 1979, 3173, 4193]
    assert periods.titles.sum() + base.release_year.isna().sum() == len(base)
    for label, mask in [
        ("Before 1980", base.release_year.lt(1980)),
        ("1980–1999", base.release_year.between(1980, 1999)),
        ("2000–2009", base.release_year.between(2000, 2009)),
        ("2010–2020", base.release_year.between(2010, 2020)),
    ]:
        cohort = base[mask.fillna(False)]
        result = periods[periods.period.eq(label)].iloc[0]
        assert len(cohort) == result.titles
        assert np.isclose(cohort.gap_pp.mean(), result.mean_gap_pp)
    cells = period_genres(base)
    direct = base[
        base.genre_list.apply(lambda g: "Documentary" in g)
        & base.release_period.eq("2010–2020").fillna(False)]

    cell = cells[cells.genre.eq("Documentary") & cells.period.eq("2010–2020")].iloc[0]
    assert len(direct) == cell.titles and np.isclose(
        direct.gap_pp.mean(), cell.mean_gap_pp)

    standardized, weights = standardized_genres(base)
    assert np.isclose(weights.sum(), 1)
    known = base.dropna(subset=["release_period"])
    for row in standardized.itertuples():
        group = known[known.genre_list.apply(lambda labels: row.genre in labels)]
        manual = sum(
            group[group.release_period.eq(period)].gap_pp.mean() * weight
            for period, weight in weights.items())

        assert np.isclose(manual, row.same_period_mix_gap_pp)
        assert np.isclose(group.gap_pp.mean(), row.observed_gap_pp)

    assert standardized_genres(base, min_per_period=20000)[0].empty
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        csv = ROOT / "data/processed/movie_scores.csv"
        (folder / csv.name).write_bytes(csv.read_bytes() + b"\n")
        (folder / "preparation.json").write_text(json.dumps(metadata))
        try:
            load_prepared(folder)
        except ValueError:
            pass
        else:
            raise AssertionError("Modified processed data were accepted")

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    stats.to_csv(output / "default_summary.csv", header=["value"])
    genre_summary(base).to_csv(output / "default_genres.csv", index=False)
    print("PASS: snapshot, quality, default counts, score formulas, genre overlap, zero scores, empty filters and CSV round-trip.")

    comparison.to_csv(output / "filter_comparison.csv", index=False)
    periods.to_csv(output / "release_period_summary.csv", index=False)
    cells.to_csv(output / "genre_period_summary.csv", index=False)
    standardized.to_csv(output / "standardized_genres.csv", index=False)
    print("PASS: raw/processed parity, integrity rejection, filter sensitivity, release-period denominators and independent standardization calculations.")
    print(stats.round(4).to_string())


if __name__ == "__main__":
    main()
