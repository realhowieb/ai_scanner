"""Backend for per-user Alpaca paper trading: crypto, account store, order client."""
from __future__ import annotations

import os
import unittest
from unittest import mock


class SecretBoxTests(unittest.TestCase):
    def setUp(self):
        os.environ["APP_ENCRYPTION_KEY"] = "unit-test-app-secret"
        import importlib

        import db.secret_box as sb
        importlib.reload(sb)
        self.sb = sb

    def test_round_trip_and_opacity(self):
        tok = self.sb.encrypt_secret("PKPAPERKEY_SECRET")
        self.assertIsNotNone(tok)
        self.assertNotIn("PKPAPER", tok)                 # not plaintext
        self.assertEqual(self.sb.decrypt_secret(tok), "PKPAPERKEY_SECRET")

    def test_bad_token_is_safe(self):
        self.assertIsNone(self.sb.decrypt_secret("not-a-valid-token"))

    def test_no_secret_disables_encryption(self):
        os.environ.pop("APP_ENCRYPTION_KEY", None)
        os.environ.pop("COOKIE_PASSWORD", None)
        import importlib

        import db.secret_box as sb
        importlib.reload(sb)
        self.assertFalse(sb.encryption_available())
        self.assertIsNone(sb.encrypt_secret("x"))


class _Cur:
    def __init__(self, fetch=None):
        self._fetch = fetch
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self._fetch

    def close(self):
        pass


class _Conn:
    def __init__(self, fetch=None):
        self.cur = _Cur(fetch)

    def cursor(self, *a, **k):
        return self.cur

    def commit(self):
        pass


class PaperAccountStoreTests(unittest.TestCase):
    def setUp(self):
        os.environ["APP_ENCRYPTION_KEY"] = "unit-test-app-secret"

    def test_save_encrypts_then_get_decrypts(self):
        import db.paper_trading as pt
        # save: capture the encrypted values written
        save_conn = _Conn()
        with mock.patch.object(pt, "_get_conn", return_value=save_conn):
            ok = pt.save_paper_account("U@x.com", "PK_LIVE", "SECRET_LIVE")
        self.assertTrue(ok)
        _sql, params = save_conn.cur.calls[0]
        key_enc, sec_enc = params[1], params[2]
        self.assertNotIn("PK_LIVE", key_enc)              # stored encrypted
        self.assertNotIn("SECRET_LIVE", sec_enc)

        # get: the stored ciphertext decrypts back to plaintext
        get_conn = _Conn(fetch={"api_key_enc": key_enc, "api_secret_enc": sec_enc})
        with mock.patch.object(pt, "_get_conn", return_value=get_conn):
            acct = pt.get_paper_account("u@x.com")
        self.assertEqual(acct, {"api_key": "PK_LIVE", "api_secret": "SECRET_LIVE"})

    def test_save_refuses_without_encryption(self):
        os.environ.pop("APP_ENCRYPTION_KEY", None)
        os.environ.pop("COOKIE_PASSWORD", None)
        import importlib

        import db.secret_box as sb
        importlib.reload(sb)
        import db.paper_trading as pt
        importlib.reload(pt)
        # No _get_conn should even be reached — refuses to store unencryptable.
        with mock.patch.object(pt, "_get_conn", side_effect=AssertionError("must not touch DB")):
            self.assertFalse(pt.save_paper_account("u", "k", "s"))


class OrderClientTests(unittest.TestCase):
    def _resp(self, code, data):
        r = mock.MagicMock()
        r.status_code = code
        r.json.return_value = data
        r.text = str(data)
        return r

    def test_market_order_success(self):
        import data.alpaca_trading as at

        req = mock.MagicMock()
        req.post.return_value = self._resp(200, {"id": "ord_1", "status": "accepted"})
        with mock.patch.object(at, "requests", req):
            out = at.submit_market_order("K", "S", "cnta", 20, "buy")
        self.assertTrue(out["ok"])
        self.assertEqual(out["order_id"], "ord_1")
        # sent a whole-share market DAY order for the upper-cased symbol
        payload = req.post.call_args.kwargs["json"]
        self.assertEqual(payload["symbol"], "CNTA")
        self.assertEqual(payload["type"], "market")
        self.assertEqual(payload["qty"], "20")

    def test_bad_qty_rejected_before_request(self):
        import data.alpaca_trading as at

        req = mock.MagicMock()
        with mock.patch.object(at, "requests", req):
            out = at.submit_market_order("K", "S", "X", 0)
        self.assertFalse(out["ok"])
        req.post.assert_not_called()

    def test_http_error_surfaces_message(self):
        import data.alpaca_trading as at

        req = mock.MagicMock()
        req.post.return_value = self._resp(403, {"message": "insufficient buying power"})
        with mock.patch.object(at, "requests", req):
            out = at.submit_market_order("K", "S", "X", 5)
        self.assertFalse(out["ok"])
        self.assertIn("insufficient buying power", out["error"])


if __name__ == "__main__":
    unittest.main()


class _FakeSt:
    """Minimal streamlit stand-in: swallows UI calls, records spinner/messages."""
    def __init__(self):
        self.errors = []
        self.successes = []
        self.session_state = {}

    def error(self, msg, *a, **k):
        self.errors.append(str(msg))

    def success(self, msg, *a, **k):
        self.successes.append(str(msg))

    def rerun(self, *a, **k):
        pass

    def spinner(self, *a, **k):
        from contextlib import nullcontext
        return nullcontext()


class SubmitAndImportTests(unittest.TestCase):
    def _row(self):
        return {"Ticker": "cnta", "Last": 12.45, "BreakoutScore": 58.0,
                "AI Confidence": 71.5}

    def test_success_imports_with_score_and_ai(self):
        import data.alpaca_trading as at
        import db.paper_trading as pt
        import db.trades as tr
        import ui.paper_trade as ptui

        logged = {}
        plan = {"stop": 11.90, "targets": [13.80, 15.0]}
        with mock.patch.object(ptui, "st", _FakeSt()), \
             mock.patch.object(pt, "get_paper_account",
                               return_value={"api_key": "K", "api_secret": "S"}), \
             mock.patch.object(at, "submit_market_order",
                               return_value={"ok": True, "order_id": "o1",
                                             "status": "filled",
                                             "filled_avg_price": 12.50}), \
             mock.patch.object(tr, "log_trade",
                               side_effect=lambda *a, **k: logged.update(
                                   {"args": a, "kw": k})):
            ptui._submit_and_import("u@x.com", self._row(), "CNTA", 20, plan)

        self.assertEqual(logged["args"][1], "CNTA")
        self.assertEqual(logged["args"][3], 20)                    # qty
        self.assertEqual(logged["args"][2], 12.50)                 # filled entry
        self.assertEqual(logged["kw"]["alpaca_order_id"], "o1")
        self.assertEqual(logged["kw"]["breakout_score"], 58.0)
        self.assertEqual(logged["kw"]["ai_confidence"], 71.5)
        self.assertEqual(logged["kw"]["stop_price"], 11.90)
        self.assertEqual(logged["kw"]["target_price"], 13.80)

    def test_rejected_order_does_not_import(self):
        import data.alpaca_trading as at
        import db.paper_trading as pt
        import db.trades as tr
        import ui.paper_trade as ptui

        fake = _FakeSt()
        with mock.patch.object(ptui, "st", fake), \
             mock.patch.object(pt, "get_paper_account",
                               return_value={"api_key": "K", "api_secret": "S"}), \
             mock.patch.object(at, "submit_market_order",
                               return_value={"ok": False, "error": "boom"}), \
             mock.patch.object(tr, "log_trade",
                               side_effect=AssertionError("must not import")):
            ptui._submit_and_import("u@x.com", self._row(), "CNTA", 20, {})
        self.assertTrue(any("boom" in e for e in fake.errors))
