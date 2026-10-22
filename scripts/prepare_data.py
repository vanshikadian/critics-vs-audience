"""Build the saved analysis table without changing the raw data."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import DATA_PATH, PROCESSED_DIR, PIPELINE_VERSION, load_movies
from scripts.check_project import audit

COLUMNS = [
    "rotten_tomatoes_link",
    "movie_title",
    "genres",
    "original_release_date",
    "release_year",
    "release_period",
    "content_rating",
    "runtime",
    "tomatometer_rating",
    "audience_rating",
    "tomatometer_count",
    "audience_count",
    "gap_pp",
    "absolute_gap_pp",
    "verdict",
    "scores_valid",
    "counts_valid",
    "comparison_eligible",
]


def main():
    source = json.loads((ROOT / "data/source.json").read_text())
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(
            "The raw CSV differs from the documented snapshot. Review the source before preparing a different version."
        )
    raw, movies = load_movies()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "movie_scores.csv"
    movies[COLUMNS].to_csv(output, index=False, lineterminator="\n")
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "source_snapshot": source["snapshot_date"],
        "raw_sha256": digest,
        "processed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(movies),
        "columns": COLUMNS,
        "quality": audit(raw),
        "paired_scores": int(movies.scores_valid.sum()),
        "comparison_eligible": int(movies.comparison_eligible.sum()),
        "missing_release_year": int(movies.release_year.isna().sum()),
        "transformations": [
            "Numeric scores, counts and runtime; no score imputation.",
            "Release year and fixed release periods from original_release_date.",
            "Audience minus critic approval and absolute gap, in percentage points.",
            "Four groups based on whether each score is at least 60%.",
            "Validity flags; all source titles retained. Filters are applied during analysis.",
        ],
    }
    (PROCESSED_DIR / "preparation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved {len(movies):,} titles to {output}; raw CSV unchanged.")


if __name__ == "__main__":
    main()
