import unittest

from scan.universe_selection import resolve_scan_universe


def _safe_call(fn, **_kwargs):
    return fn()


def _identity(symbols):
    return list(symbols or [])


def _sanitize(symbols):
    out = []
    for sym in symbols or []:
        sym = str(sym).strip().upper()
        if sym:
            out.append(sym)
    return out


class UniverseSelectionTests(unittest.TestCase):
    def test_sp500_loads_and_caches_filtered_symbols(self):
        state = {}

        result = resolve_scan_universe(
            "SP500",
            state,
            is_admin=False,
            safe_call=_safe_call,
            load_sp500_universe=lambda: [" aapl ", "msft"],
            load_nasdaq_universe=lambda: ["qqq"],
            filter_universe=_identity,
            sanitize_symbols=_sanitize,
        )

        self.assertEqual(result, ["AAPL", "MSFT"])
        self.assertEqual(state["sp500_universe"], ["AAPL", "MSFT"])
        self.assertNotIn("nasdaq_universe", state)

    def test_nasdaq_respects_non_admin_cap(self):
        state = {"max_nasdaq_scan": 2}

        result = resolve_scan_universe(
            "NASDAQ",
            state,
            is_admin=False,
            safe_call=_safe_call,
            load_sp500_universe=lambda: [],
            load_nasdaq_universe=lambda: ["a", "b", "c"],
            filter_universe=_identity,
            sanitize_symbols=_sanitize,
        )

        self.assertEqual(result, ["A", "B"])
        self.assertEqual(state["nasdaq_capped"], ["A", "B"])

    def test_nasdaq_admin_gets_full_loaded_universe(self):
        state = {"max_nasdaq_scan": 2}

        result = resolve_scan_universe(
            "NASDAQ",
            state,
            is_admin=True,
            safe_call=_safe_call,
            load_sp500_universe=lambda: [],
            load_nasdaq_universe=lambda: ["a", "b", "c"],
            filter_universe=_identity,
            sanitize_symbols=_sanitize,
        )

        self.assertEqual(result, ["A", "B", "C"])

    def test_combo_uses_sp500_plus_capped_nasdaq_then_combo_cap(self):
        state = {"max_nasdaq_scan": 2, "max_combo_scan": 3}

        result = resolve_scan_universe(
            "COMBO",
            state,
            is_admin=False,
            safe_call=_safe_call,
            load_sp500_universe=lambda: ["spy"],
            load_nasdaq_universe=lambda: ["a", "b", "c"],
            filter_universe=_identity,
            sanitize_symbols=_sanitize,
        )

        self.assertEqual(result, ["SPY", "A", "B"])
        self.assertEqual(state["combo_capped"], ["SPY", "A", "B"])

    def test_combo_admin_expands_to_full_combo(self):
        state = {"max_nasdaq_scan": 1, "max_combo_scan": 2}

        result = resolve_scan_universe(
            "COMBO",
            state,
            is_admin=True,
            safe_call=_safe_call,
            load_sp500_universe=lambda: ["spy"],
            load_nasdaq_universe=lambda: ["a", "b", "c"],
            filter_universe=_identity,
            sanitize_symbols=_sanitize,
        )

        self.assertEqual(result, ["SPY", "A", "B", "C"])

    def test_existing_cache_is_reused(self):
        state = {"sp500_universe": ["CACHED"]}

        result = resolve_scan_universe(
            "SP500",
            state,
            is_admin=False,
            safe_call=_safe_call,
            load_sp500_universe=lambda: ["new"],
            load_nasdaq_universe=lambda: [],
            filter_universe=_identity,
            sanitize_symbols=_sanitize,
        )

        self.assertEqual(result, ["CACHED"])

    def test_nasdaq_cap_change_refreshes_cached_cap(self):
        state = {"max_nasdaq_scan": 1}
        kwargs = {
            "is_admin": False,
            "safe_call": _safe_call,
            "load_sp500_universe": lambda: [],
            "load_nasdaq_universe": lambda: ["a", "b", "c"],
            "filter_universe": _identity,
            "sanitize_symbols": _sanitize,
        }

        first = resolve_scan_universe("NASDAQ", state, **kwargs)
        state["max_nasdaq_scan"] = 3
        second = resolve_scan_universe("NASDAQ", state, **kwargs)

        self.assertEqual(first, ["A"])
        self.assertEqual(second, ["A", "B", "C"])
        self.assertEqual(state["nasdaq_capped_limit"], 3)

    def test_combo_cap_change_refreshes_cached_cap(self):
        state = {"max_nasdaq_scan": 3, "max_combo_scan": 2}
        kwargs = {
            "is_admin": False,
            "safe_call": _safe_call,
            "load_sp500_universe": lambda: ["spy"],
            "load_nasdaq_universe": lambda: ["a", "b", "c"],
            "filter_universe": _identity,
            "sanitize_symbols": _sanitize,
        }

        first = resolve_scan_universe("COMBO", state, **kwargs)
        state["max_combo_scan"] = 4
        second = resolve_scan_universe("COMBO", state, **kwargs)

        self.assertEqual(first, ["SPY", "A"])
        self.assertEqual(second, ["SPY", "A", "B", "C"])
        self.assertEqual(state["combo_capped_limit"], 4)

    def test_combo_transform_runs_before_final_combo_cap(self):
        state = {"max_nasdaq_scan": 3, "max_combo_scan": 2}

        result = resolve_scan_universe(
            "COMBO",
            state,
            is_admin=False,
            safe_call=_safe_call,
            load_sp500_universe=lambda: ["spy"],
            load_nasdaq_universe=lambda: ["a", "b", "c"],
            filter_universe=_identity,
            sanitize_symbols=_sanitize,
            combo_universe_transform=lambda symbols: [sym for sym in symbols if sym != "SPY"],
            combo_cache_key=("liquidity", 1),
        )

        self.assertEqual(result, ["A", "B"])

    def test_combo_transform_cache_key_refreshes_cached_cap(self):
        state = {"max_nasdaq_scan": 3, "max_combo_scan": 2}
        kwargs = {
            "is_admin": False,
            "safe_call": _safe_call,
            "load_sp500_universe": lambda: ["spy"],
            "load_nasdaq_universe": lambda: ["a", "b", "c"],
            "filter_universe": _identity,
            "sanitize_symbols": _sanitize,
        }

        first = resolve_scan_universe(
            "COMBO",
            state,
            combo_universe_transform=lambda symbols: symbols,
            combo_cache_key=("liquidity", 1),
            **kwargs,
        )
        second = resolve_scan_universe(
            "COMBO",
            state,
            combo_universe_transform=lambda symbols: [sym for sym in symbols if sym != "SPY"],
            combo_cache_key=("liquidity", 2),
            **kwargs,
        )

        self.assertEqual(first, ["SPY", "A"])
        self.assertEqual(second, ["A", "B"])
        self.assertEqual(state["combo_capped_transform_key"], ("liquidity", 2))


if __name__ == "__main__":
    unittest.main()
