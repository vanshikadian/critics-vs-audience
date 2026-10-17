"""Check the dataset, calculations, filters, and CSV export."""

import hashlib
import io
import json
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
        base.gap_pp, base.audience_rating - base.tomatometer_rating
    )
    assert base.rotten_tomatoes_link.is_unique
    assert base.verdict.value_counts().to_dict() == dict(
        zip(GROUPS, [5122, 3069, 1123, 1189])
    )
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
    assert len(
        pd.read_csv(io.StringIO(export.to_csv(index=False)))
    ) == len(base)
    assert export.rotten_tomatoes_link.is_unique
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    stats.to_csv(output / "default_summary.csv", header=["value"])
    genre_summary(base).to_csv(output / "default_genres.csv", index=False)
    print(
        "PASS: snapshot, quality, default counts, score formulas, genre overlap, zero scores, empty filters and CSV round-trip."
    )
    print(stats.round(4).to_string())


if __name__ == "__main__":
    main()
