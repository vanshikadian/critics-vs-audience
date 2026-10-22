"""Exercise every analysis view and the CSV payloads served by Streamlit."""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import PROFILES


def main():
    downloads = {}
    original = MemoryMediaFileStorage.load_and_get_id

    def capture(self, data, mimetype, kind, filename=None):
        if filename and filename.startswith("split_decision_"):
            downloads[filename] = data
        return original(self, data, mimetype, kind, filename)

    def healthy(app):
        assert not app.exception, [error.message for error in app.exception]

    def slider(app, label, value):
        next(item for item in app.slider if item.label == label).set_value(value).run()
        healthy(app)

    def csv(name):
        return pd.read_csv(io.BytesIO(downloads[f"split_decision_{name}.csv"]))

    with patch.object(MemoryMediaFileStorage, "load_and_get_id", capture):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        healthy(app)
        assert [item.label for item in app.tabs] == [
            "Overview",
            "Genres",
            "Filter test",
            "Release periods",
            "Data & methods",
        ]
        assert [item.value for item in app.metric] == ["10,503", "22.0%", "15.2 pp"]
        assert len(csv("titles")) == 10503
        assert (
            csv("filter_comparison").set_index("genre").loc["Horror", "direction"]== "Direction changes")
        assert len(csv("release_periods")) > 0 and len(csv("period_mix")) > 0
        app.radio[0].set_value("Median gap").run()
        healthy(app)
        app.radio[0].set_value("Mean gap").run()
        healthy(app)
        for preset in PROFILES:
            app.selectbox[1].set_value(preset).run()
            healthy(app)
            exported = csv("filter_comparison")
            assert exported.titles_B.ge(0).all()
            if preset == app.selectbox[0].value:
                assert exported.change_pp.eq(0).all()
                assert any("same rule" in item.value for item in app.info)
        app.selectbox[0].set_value("Strict · 50 / 1,000").run()

        healthy(app)
        assert (csv("filter_comparison").set_index("genre").loc["Horror", "titles_A"] == 553)
        app.selectbox[0].set_value("Default · 20 / 100").run()
        app.selectbox[1].set_value("Strict · 50 / 1,000").run()
        app.multiselect[0].set_value(["Horror"]).run()

        healthy(app)
        assert app.metric[0].value == "1,062"
        assert csv("filter_comparison").genre.tolist() == ["Horror"]
        app.select_slider[0].set_value(1000).run()
        slider(app, "Minimum critic reviews", 50)

        exported = csv("titles")
        assert len(exported) == 553 and exported.gap_pp.mean() > 0
        assert (exported.tomatometer_count.ge(50).all() and exported.audience_count.ge(1000).all())
        app.text_input[0].set_value("Alien").run()

        healthy(app)
        assert (
            len(csv("titles")) > 0
            and csv("titles") .movie_title.str.contains("Alien", case=False, regex=False) .all())
        app.text_input[0].set_value("no such title xyz").run()

        healthy(app)
        assert csv("titles").empty
        app.text_input[0].set_value("").run()
        app.multiselect[0].set_value(["Western"]).run()

        slider(app, "Minimum titles per displayed genre", 500)
        assert any("No genre has enough" in item.value for item in app.info)
        assert any("No genre meets" in item.value for item in app.info)
        slider(app, "Minimum titles per genre and release period", 100)
        assert csv("period_mix").empty
        slider(app, "Minimum critic reviews", 600)
        assert not app.metric and csv("titles").empty
        assert any("No titles match" in item.value for item in app.info)
        assert any("No titles with known" in item.value for item in app.info)
        assert len(csv("filter_comparison")) > 0
    print("PASS: five views, default metrics, all filter presets, mean/median, genre and count filters, "
        "release-period minimums, search, empty states and all four CSV payloads.")


if __name__ == "__main__":
    main()
