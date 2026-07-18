"""The track-record badge shows an accumulating state instead of vanishing."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_DEPS = (
    importlib.util.find_spec("streamlit") is not None
    and importlib.util.find_spec("pandas") is not None
)


@unittest.skipUnless(_DEPS, "result_helpers needs streamlit + pandas")
class TrackRecordBadgeTests(unittest.TestCase):
    def _render(self, tr_by_horizon):
        import ui.result_helpers as h

        fake_st = mock.MagicMock()
        # st.columns(n) must yield n column mocks that also support .metric.
        fake_st.columns.side_effect = lambda n: [mock.MagicMock() for _ in range(n)]

        def fake_cached(horizon_days=5):
            return tr_by_horizon.get(horizon_days)

        with mock.patch.object(h, "st", fake_st), \
             mock.patch.object(h, "_cached_track_record", side_effect=fake_cached):
            h.render_track_record_badge()
        return fake_st

    def test_thin_data_shows_building_progress_not_full_badge(self):
        thin = {"sample_size": 40, "runs_used": 3, "avg_return": -0.02,
                "win_rate": 0.3, "horizon_days": 5}
        st = self._render({5: thin})
        st.progress.assert_called_once()          # accumulating state rendered
        st.metric.assert_not_called()             # not the full hero badge

    def test_no_data_renders_nothing(self):
        st = self._render({5: {"sample_size": 0}})
        st.progress.assert_not_called()
        st.metric.assert_not_called()

    def test_sufficient_data_shows_full_badge(self):
        full = {"sample_size": 200, "runs_used": 12, "avg_return": 0.015,
                "win_rate": 0.55, "horizon_days": 5, "benchmark": "SPY", "top_n": 5}
        st = self._render({5: full})
        st.progress.assert_not_called()           # no building bar
        self.assertTrue(st.columns.called)         # hero tiles laid out


if __name__ == "__main__":
    unittest.main()
