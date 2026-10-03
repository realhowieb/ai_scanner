"""Admin research heatmap: a group with no horizon (pending observations) must not
become a NaN column label. Streamlit writes labels into the Arrow table's JSON
metadata and the browser failed with 'JSON Parse error: Unexpected identifier NaN'."""
import importlib.util
import unittest
from unittest import mock

HAS_ST = importlib.util.find_spec("streamlit") is not None and importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(HAS_ST, "needs streamlit + pyarrow")
class HeatmapLabelTests(unittest.TestCase):
    def render(self, groups, metric="Observation Count"):
        from ui import admin_analytics as aa

        frames = []
        with mock.patch.object(aa.st, "dataframe", side_effect=lambda df, **k: frames.append(df)), \
                mock.patch.object(aa.st, "caption"), mock.patch.object(aa.st, "info") as info:
            aa._render_heatmap(groups, row_key="signal", col_key="horizon", metric_label=metric, min_n=30)
        return frames, info

    def arrow_metadata(self, frame):
        import pyarrow as pa
        from streamlit.dataframe_util import convert_pandas_df_to_arrow_bytes

        table = pa.ipc.open_stream(convert_pandas_df_to_arrow_bytes(frame)).read_all()
        return table.schema.metadata[b"pandas"].decode()

    def test_pending_placeholder_groups_are_dropped(self):
        groups = [{"signal": "breakout", "horizon": "+1d", "n": 40, "median_return": 0.01},
                  {"signal": "breakout", "horizon": None, "n": 0, "median_return": None},
                  {"signal": "prebreakout", "horizon": None, "n": 0, "median_return": None}]
        frames, _ = self.render(groups)
        self.assertEqual(list(frames[0].columns), ["+1d"])
        self.assertNotIn("NaN", self.arrow_metadata(frames[0]))

    def test_missing_key_with_data_gets_a_label_not_nan(self):
        groups = [{"signal": "breakout", "horizon": "+1d", "n": 40, "median_return": 0.01},
                  {"signal": None, "horizon": None, "n": 35, "median_return": 0.02}]
        for metric in ("Observation Count", "Median Return"):
            frames, _ = self.render(groups, metric)
            self.assertIn("(none)", list(frames[0].columns))
            self.assertNotIn("NaN", self.arrow_metadata(frames[0]))

    def test_only_pending_rows_shows_the_empty_message(self):
        frames, info = self.render([{"signal": "breakout", "horizon": None, "n": 0}])
        self.assertEqual(frames, [])
        info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
