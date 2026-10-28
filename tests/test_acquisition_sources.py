"""Sign-up source labels (ui/acquisition.classify_source) and their admin names."""
import unittest


class SourceClassificationTests(unittest.TestCase):
    def test_channels(self):
        from ui.acquisition import attribution_from_params, classify_source

        cases = [
            ({}, None, "direct"),
            ({"ref": "abcd2345", "utm_source": "reddit"}, None, "invite"),
            ({"utm_source": "twitter"}, None, "x"),
            ({}, "https://t.co/abc", "x"),
            ({}, "https://old.reddit.com/r/pennystocks/", "reddit"),
            ({}, "https://www.google.co.uk/", "google"),
            ({}, "https://notgoogle.example/", "other_unknown"),
            ({"utm_source": "YouTube"}, None, "youtube"),
            ({}, "https://discord.gg/xyz", "discord"),
            ({"utm_source": "stocktwits"}, None, "stocktwits"),
            ({"utm_source": "ig"}, None, "facebook"),
            ({}, "https://www.linkedin.com/feed/", "linkedin"),
            ({"utm_source": "newsletter"}, None, "other_unknown"),
        ]
        for params, ref, want in cases:
            self.assertEqual(classify_source(params, ref), want, (params, ref))
        # An invite code is not a referrer site.
        self.assertNotIn("referrer_domain", attribution_from_params({"ref": "abcd2345"}))

    def test_every_source_has_an_admin_label(self):
        from api.admin_analytics import SOURCE_LABELS
        from ui.acquisition import SOURCE_NAMES

        self.assertEqual(set(SOURCE_NAMES), set(SOURCE_LABELS))


if __name__ == "__main__":
    unittest.main()
