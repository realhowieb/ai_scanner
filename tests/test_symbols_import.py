"""Guard: data.symbols imports (no variable-width lookbehind) + class-share norm."""
import unittest


class SymbolsImportTests(unittest.TestCase):
    def test_module_imports(self):
        import data.symbols  # must not raise on Python 3.11+
        self.assertTrue(hasattr(data.symbols, "_replace_class_separator_to_dash"))

    def test_class_separator_to_dash(self):
        from data.symbols import _replace_class_separator_to_dash as f
        self.assertEqual(f("BRK.B"), "BRK-B")
        self.assertEqual(f("BF.B"), "BF-B")
        self.assertEqual(f("BRK B"), "BRK-B")
        self.assertEqual(f("AAPL"), "AAPL")        # no separator, unchanged
        self.assertEqual(f("ABC.DEF"), "ABC.DEF")  # 3-char tail is not a class share


if __name__ == "__main__":
    unittest.main()
