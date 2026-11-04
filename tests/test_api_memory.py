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


    def test_container_memory_subtracts_inactive_file_cache(self):
        files = {"/sys/fs/cgroup/memory.current": "524288000\n",
                 "/sys/fs/cgroup/memory.stat": "anon 1\ninactive_file 104857600\nactive_file 5\n"}

        def fake_open(path, mode="r"):
            if path not in files:
                raise OSError(path)
            data = files[path]
            return mock.mock_open(read_data=data.encode() if "b" in mode else data)()

        with mock.patch("builtins.open", side_effect=fake_open):
            self.assertEqual(memory.container_mb(), 400.0)

    def test_warning_uses_container_peak_when_present(self):
        with mock.patch.object(memory, "rss_mb", return_value=430.0), \
             mock.patch.object(memory, "peak_rss_mb", return_value=435.8), \
             mock.patch.object(memory, "container_mb", return_value=360.0), \
             mock.patch.object(memory, "peak_container_mb", return_value=380.0):
            status = memory.memory_status()
        self.assertFalse(status["memory_warning"])
        self.assertEqual(status["container_peak_mb"], 380.0)
        with mock.patch.object(memory, "rss_mb", return_value=430.0), \
             mock.patch.object(memory, "peak_rss_mb", return_value=435.8), \
             mock.patch.object(memory, "container_mb", return_value=None), \
             mock.patch.object(memory, "peak_container_mb", return_value=None):
            self.assertTrue(memory.memory_status()["memory_warning"])





class MemoryMaintenanceTests(unittest.TestCase):
    def test_current_pressure_trims_free_pages_and_reports_before_and_after(self):
        from api.today import TTLCache

        lib = mock.Mock()
        lib.malloc_trim.return_value = 1
        with mock.patch.object(memory, "container_mb", side_effect=[455.0, 310.0]), \
                mock.patch.object(TTLCache, "prune_all", return_value=3), \
                mock.patch.object(TTLCache, "statistics", return_value={"entries": 4}), \
                mock.patch.object(memory, "_glibc", return_value=lib), mock.patch("gc.collect") as collect:
            out = memory.maintain_memory()
        self.assertEqual((out["before_mb"], out["after_mb"], out["expired_cache_entries"]), (455, 310, 3))
        lib.malloc_trim.assert_called_once_with(0)
        collect.assert_called_once()

    def test_low_current_usage_does_not_trim_because_of_an_old_peak(self):
        from api.today import TTLCache

        with mock.patch.object(memory, "container_mb", return_value=200.0), \
                mock.patch.object(memory, "peak_container_mb", return_value=490.0), \
                mock.patch.object(TTLCache, "prune_all", return_value=0), \
                mock.patch.object(memory, "_glibc") as libc, mock.patch("gc.collect") as collect:
            memory.maintain_memory()
        libc.assert_not_called()
        collect.assert_not_called()

    def test_disabled_is_a_noop(self):
        with mock.patch.dict(os.environ, {"HSF_MEMORY_MAINTENANCE_ENABLED": "0"}), \
                mock.patch.object(memory, "container_mb") as sample:
            self.assertEqual(memory.maintain_memory(), {"disabled": True})
        sample.assert_not_called()

    def test_growth_is_logged_even_when_the_operation_fails(self):
        with mock.patch.object(memory, "rss_mb", side_effect=[200.0, 235.0]), \
                self.assertLogs("hsf.memory", level="INFO") as logs:
            with self.assertRaises(RuntimeError), memory.observe_memory("custom_scan"):
                raise RuntimeError("provider failed")
        self.assertIn('"rss_delta_mb": 35.0', logs.output[0])


if __name__ == "__main__":
    unittest.main()
