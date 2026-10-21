"""Today and the latest scan revalidate to 304 although each answer carries the request time."""
import unittest

from api.etag import etag_for, hash_view


def _tag(path, body):
    return etag_for(hash_view(path, body))


class ClockStampedETagTests(unittest.TestCase):
    def test_request_time_fields_do_not_change_the_tag(self):
        a = b'{"as_of":"2026-10-21T16:22:29.4+00:00","market":{"freshness":{"checked_at":"2026-10-21T16:22:29.4+00:00"}},"x":1}'
        b = b'{"as_of":"2026-10-21T16:40:02.1+00:00","market":{"freshness":{"checked_at":"2026-10-21T16:40:02.1+00:00"}},"x":1}'
        self.assertEqual(_tag("/v1/today", a), _tag("/v1/today", b))
        s1 = b'{"rows":[1],"freshness":{"checked_at":"2026-10-21T16:22:32+00:00"}}'
        s2 = b'{"rows":[1],"freshness":{"checked_at":"2026-10-21T16:22:35+00:00"}}'
        self.assertEqual(_tag("/v1/scans/latest", s1), _tag("/v1/scans/latest", s2))

    def test_content_and_the_hour_still_change_the_tag(self):
        a = b'{"as_of":"2026-10-21T16:59:00+00:00","x":1}'
        self.assertNotEqual(_tag("/v1/today", a), _tag("/v1/today", b'{"as_of":"2026-10-21T17:00:01+00:00","x":1}'))
        self.assertNotEqual(_tag("/v1/today", a), _tag("/v1/today", b'{"as_of":"2026-10-21T16:59:00+00:00","x":2}'))

    def test_other_paths_hash_the_whole_body(self):
        a = b'{"checked_at":"2026-10-21T16:22:32+00:00"}'
        b = b'{"checked_at":"2026-10-21T16:22:35+00:00"}'
        self.assertNotEqual(_tag("/v1/admin/providers", a), _tag("/v1/admin/providers", b))


if __name__ == "__main__":
    unittest.main()
