"""utils/st_compat.py: the stand-in for Streamlit's cache that shared helpers still use."""
import unittest

from utils.st_compat import st


class CacheDataTests(unittest.TestCase):
    def test_caches_by_argument_ignores_underscore_args_and_hands_out_copies(self):
        calls = []

        @st.cache_data(ttl=60, max_entries=2, show_spinner=False)
        def load(x, _conn=None):
            calls.append(x)
            return {"rows": [x]}

        first = load(1, _conn=object())
        first["rows"].append(99)                      # a caller mutating its copy...
        self.assertEqual(load(1, _conn=object()), {"rows": [1]})  # ...never changes the cache
        self.assertEqual(calls, [1])
        load(2), load(3), load(1)                     # max_entries=2 evicts the oldest
        self.assertEqual(calls, [1, 2, 3, 1])
        load.clear()
        load(3)
        st.cache_data.clear()
        load(3)
        self.assertEqual(calls, [1, 2, 3, 1, 3, 3])

    def test_ttl_expires(self):
        calls = []

        @st.cache_data(ttl=0)
        def now(x):
            calls.append(x)
            return x

        now(1), now(1)
        self.assertEqual(calls, [1, 1])

    def test_ui_calls_are_harmless(self):
        a, b = st.columns(2)
        with a:
            st.markdown("x")
        self.assertEqual(st.secrets, {})
        st.session_state["k"] = 1
        self.assertEqual(st.session_state.k, 1)


if __name__ == "__main__":
    unittest.main()
