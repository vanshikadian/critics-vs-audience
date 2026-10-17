"""Exercise the app controls and the CSV payload served by Streamlit."""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    downloads = []
    original = MemoryMediaFileStorage.load_and_get_id

    def capture(self, data, mimetype, kind, filename=None):
        if filename == "split_decision_titles.csv":
            downloads.append(data)
        return original(self, data, mimetype, kind, filename)

    def healthy(app):
        assert not app.exception, [error.message for error in app.exception]

    with patch.object(MemoryMediaFileStorage, "load_and_get_id", capture):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        healthy(app)
        assert [m.value for m in app.metric] == ["10,503", "22.0%", "15.2 pp"]
        assert len(app.tabs) == 2
        assert len(pd.read_csv(io.BytesIO(downloads[-1]))) == 10503
        app.multiselect[0].set_value(["Horror"]).run()
        healthy(app)
        assert app.metric[0].value == "1,062"
        app.select_slider[0].set_value(1000).run()
        next(s for s in app.slider if s.label == "Minimum critic reviews").set_value(50).run()
        healthy(app)
        exported = pd.read_csv(io.BytesIO(downloads[-1]))
        assert exported.tomatometer_count.ge(50).all()
        assert exported.audience_count.ge(1000).all()
        assert exported.genres.str.contains("Horror", regex=False).all()
        assert exported.gap_pp.mean() > 0
        app.text_input[0].set_value("no such title xyz").run()
        healthy(app)
        assert pd.read_csv(io.BytesIO(downloads[-1])).empty
        app.text_input[0].set_value("Alien").run()
        healthy(app)
        exported = pd.read_csv(io.BytesIO(downloads[-1]))
        assert len(exported) > 0
        assert exported.movie_title.str.contains("Alien", case=False, regex=False).all()
        app.text_input[0].set_value("").run()
        app.multiselect[0].set_value(["Western"]).run()
        next(
            s for s in app.slider if s.label == "Minimum titles per displayed genre"
        ).set_value(500).run()
        healthy(app)
        assert any("No genre has enough" in item.value for item in app.info)
        next(s for s in app.slider if s.label == "Minimum critic reviews").set_value(600).run()
        healthy(app)
        assert not app.metric
        assert any("No titles match" in item.value for item in app.info)
    print("PASS: both tabs rendered; baseline metrics; genre, review-count and genre-size controls; title search; empty states; filtered/search CSV payloads.")


if __name__ == "__main__":
    main()
