"""fetch_alpaca_snapshots must map class shares (BRK-B) so one symbol can't
400 the whole batch."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_STREAMLIT = importlib.util.find_spec("streamlit") is not None


class _Resp:
    def __init__(self, code, data):
        self.status_code = code
        self._data = data

    def json(self):
        return self._data


@unittest.skipUnless(_STREAMLIT, "market_data imports streamlit")
class SnapshotClassShareTests(unittest.TestCase):
    def setUp(self):
        import market_data as m

        try:
            m.fetch_alpaca_snapshots.clear()  # drop cached results between tests
        except Exception:
            pass

    def test_class_share_sent_as_dot_and_mapped_back(self):
        import market_data as m

        sent = {}

        def fake_get(url, headers=None, params=None, timeout=None):
            syms = params["symbols"].split(",")
            sent["syms"] = syms
            sent["feed"] = params.get("feed")
            # Dash form would 400 the whole batch; dot form is accepted.
            if any("-" in s for s in syms):
                return _Resp(400, {})
            return _Resp(200, {s: {"latestTrade": {"p": 100.0}} for s in syms})

        with mock.patch.object(m, "_get_alpaca_headers", return_value={"k": "v"}), \
             mock.patch.object(m, "_get_alpaca_base_urls", return_value={"data_url": "https://x"}), \
             mock.patch.object(m, "_get_alpaca_data_feed", return_value="iex"), \
             mock.patch.object(m, "requests", mock.MagicMock(get=fake_get)):
            out = m.fetch_alpaca_snapshots(["AMD", "BRK-B", "MSFT"])

        self.assertIn("BRK.B", sent["syms"])          # dot form sent
        self.assertNotIn("BRK-B", sent["syms"])
        self.assertEqual(sent.get("feed"), "iex")
        self.assertEqual(set(out.keys()), {"AMD", "BRK-B", "MSFT"})  # mapped back
        self.assertTrue(out["AMD"])                   # batch not poisoned

    def test_snapshot_feed_is_explicit(self):
        import market_data as m

        sent = {}

        def fake_get(url, headers=None, params=None, timeout=None):
            sent.update(params or {})
            return _Resp(200, {"AMD": {"latestTrade": {"p": 100.0}}})

        with mock.patch.object(m, "_get_alpaca_headers", return_value={"k": "v"}), \
             mock.patch.object(m, "_get_alpaca_base_urls", return_value={"data_url": "https://x"}), \
             mock.patch.object(m, "_get_alpaca_data_feed", return_value="sip"), \
             mock.patch.object(m, "requests", mock.MagicMock(get=fake_get)):
            out = m.fetch_alpaca_snapshots(["AMD"])

        self.assertEqual(sent["feed"], "sip")
        self.assertIn("AMD", out)


if __name__ == "__main__":
    unittest.main()
