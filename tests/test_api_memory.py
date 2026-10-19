import os
import sys
import unittest
from unittest import mock

from api import memory


class AllocatorTuningTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux") and memory._glibc() is not None, "glibc only")
    def test_caps_arenas_at_two_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HSF_MALLOC_ARENA_MAX", None)
            self.assertEqual(memory.tune_allocator(), 2)

    def test_zero_leaves_the_allocator_alone(self):
        lib = mock.Mock()
        with mock.patch.dict(os.environ, {"HSF_MALLOC_ARENA_MAX": "0"}), \
             mock.patch.object(memory, "_glibc", return_value=lib):
            self.assertIsNone(memory.tune_allocator())
        lib.mallopt.assert_not_called()

    def test_bad_value_falls_back_to_two(self):
        lib = mock.Mock()
        lib.mallopt.return_value = 1
        with mock.patch.dict(os.environ, {"HSF_MALLOC_ARENA_MAX": "lots"}), \
             mock.patch.object(memory, "_glibc", return_value=lib):
            self.assertEqual(memory.tune_allocator(), 2)
        lib.mallopt.assert_called_once_with(memory.M_ARENA_MAX, 2)

    def test_no_glibc_is_a_no_op(self):
        with mock.patch.object(memory, "_glibc", return_value=None):
            self.assertIsNone(memory.tune_allocator())

    @unittest.skipUnless(os.path.exists("/proc/self/statm"), "Linux only")
    def test_rss_is_reported_in_mb(self):
        rss = memory.rss_mb()
        self.assertIsInstance(rss, float)
        self.assertGreater(rss, 1)


if __name__ == "__main__":
    unittest.main()
