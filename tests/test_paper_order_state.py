"""Tracking transitions reject delayed broker responses without losing newer fills."""
import unittest

from domain.paper_order_state import transition


class OrderStateTests(unittest.TestCase):
    def test_saved_terminal_status_and_larger_partial_fill_survive_old_responses(self):
        for old, new in (({"filled_qty": "2", "status": "filled"}, {"filled_qty": "1", "status": "partially_filled"}),
                         ({"filled_qty": "1", "status": "partially_filled"}, {"filled_qty": "0", "status": "accepted"}),
                         ({"filled_qty": "1", "status": "canceled"}, {"filled_qty": "1", "status": "partially_filled"})):
            with self.subTest(old=old, new=new):
                self.assertEqual(transition({"state": "submitted", "result": old}, "submitted", new),
                                 ("submitted", old, None))

    def test_touch_or_rejection_cannot_reset_an_existing_submission(self):
        current = {"state": "submitted", "result": {"status": "accepted"}}
        for state in ("pending", "rejected"):
            self.assertEqual(transition(current, state), ("submitted", current["result"], None))

    def test_completed_receipt_is_immutable(self):
        old = {"state": "synced", "result": {"status": "filled", "filled_qty": "2"}}
        self.assertEqual(transition(old, "submitted", {"status": "accepted"}), ("synced", old["result"], None))
