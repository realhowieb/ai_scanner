"""Six integrations: Render Key Value (shared rate limits/cache), Alpaca News, SEC EDGAR
filings + INSIDER_BUY alerts, Telegram/Discord alert delivery, Cloudflare Turnstile and
symbolicated web errors for Sentry. No network: every provider call is faked."""
import time
import unittest
from unittest import mock

from tests.test_api_account import AccountApiTestCase
from tests.test_api_v1 import DEPS


# ---- Render Key Value --------------------------------------------------------------------------------
class FakeRedis:
    """Enough of redis.Redis for shared_cache and the rate-limit script."""

    def __init__(self):
        self.kv, self.zsets = {}, {}

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v, ex=None):
        self.kv[k] = v

    def ping(self):
        return True

    def eval(self, script, nkeys, key, now, window, limit, member):
        z = self.zsets.setdefault(key, {})
        for m in [m for m, s in z.items() if s <= now - window]:
            del z[m]
        if len(z) >= limit:
            oldest = min(z.values())
            return [0, str(int(oldest + window - now) + 1)]
        z[member] = now
        return [1, "0"]


class BrokenRedis(FakeRedis):
    def get(self, k):
        raise ConnectionError("down")

    def eval(self, *a, **k):
        raise ConnectionError("down")


class SharedCacheTests(unittest.TestCase):
    def setUp(self):
        from api import ratelimit, shared_cache

        shared_cache.reset()
        ratelimit.reset()
        self.addCleanup(shared_cache.reset)
        self.addCleanup(ratelimit.reset)

    def test_without_redis_url_values_live_in_memory(self):
        from api import shared_cache

        with mock.patch.dict("os.environ", {"REDIS_URL": ""}):
            self.assertFalse(shared_cache.configured())
            calls = []
            v1 = shared_cache.cached("k", lambda: calls.append(1) or {"a": 1}, 60)
            v2 = shared_cache.cached("k", lambda: calls.append(1) or {"a": 2}, 60)
        self.assertEqual((v1, v2, len(calls)), ({"a": 1}, {"a": 1}, 1))

    def test_redis_stores_json_and_rate_limits_are_shared(self):
        from fastapi import HTTPException

        from api import ratelimit, shared_cache

        fake = FakeRedis()
        with mock.patch.dict("os.environ", {"REDIS_URL": "redis://red-test:6379"}), \
                mock.patch.object(shared_cache, "client", return_value=fake):
            shared_cache.put("x", [1, 2], 30)
            self.assertEqual(fake.kv["hsf:x"], "[1,2]")
            self.assertEqual(shared_cache.get("x"), [1, 2])
            for _ in range(5):
                ratelimit.check("signup", "203.0.113.9")
            with self.assertRaises(HTTPException) as e:
                ratelimit.check("signup", "203.0.113.9")
        self.assertEqual(e.exception.status_code, 429)
        self.assertTrue(int(e.exception.headers["Retry-After"]) >= 1)
        key = next(iter(fake.zsets))
        self.assertNotIn("203.0.113.9", key)  # callers are hashed
        self.assertEqual(ratelimit._hits, {})  # counted in Key Value, not memory

    def test_key_value_outage_falls_back_to_memory(self):
        from api import ratelimit, shared_cache

        with mock.patch.dict("os.environ", {"REDIS_URL": "redis://red-test:6379"}), \
                mock.patch.object(shared_cache, "client", return_value=BrokenRedis()):
            ratelimit.check("login", "198.51.100.1")
            self.assertEqual(shared_cache.get("missing"), None)
        self.assertEqual(len(ratelimit._hits), 1)

    def test_status_reports_ping(self):
        from api import shared_cache

        with mock.patch.dict("os.environ", {"REDIS_URL": "redis://red-test:6379"}), \
                mock.patch.object(shared_cache, "client", return_value=FakeRedis()):
            s = shared_cache.status()
        self.assertTrue(s["configured"])
        self.assertTrue(s["ok"])


# ---- Alpaca News -------------------------------------------------------------------------------------
NEWS = {"news": [
    {"id": 1, "headline": "Apple <b>beats</b> estimates", "summary": "", "source": "benzinga", "author": "A",
     "url": "https://www.benzinga.com/x", "created_at": "2026-10-21T20:05:00Z", "symbols": ["AAPL", "MSFT"]},
    {"id": 2, "headline": "Bad link", "url": "javascript:alert(1)", "created_at": "2026-10-21T19:00:00Z",
     "symbols": ["AAPL"]},
    {"id": 3, "headline": "", "created_at": "2026-10-21T18:00:00Z", "symbols": ["AAPL"]},
]}


class NewsTests(unittest.TestCase):
    def setUp(self):
        from api import shared_cache

        shared_cache.reset()
        self.addCleanup(shared_cache.reset)

    def test_stock_news_cleans_items_and_is_shared(self):
        from api import news

        with mock.patch.object(news, "_fetch", return_value=NEWS) as f:
            a = news.stock_news("aapl")
            b = news.stock_news("AAPL")
        self.assertEqual(f.call_count, 1)
        self.assertEqual(a, b)
        self.assertTrue(a["available"])
        self.assertEqual([i["headline"] for i in a["items"]], ["Apple beats estimates", "Bad link"])
        self.assertIsNone(a["items"][1]["url"])
        self.assertEqual(f.call_args.args[0]["symbols"], "AAPL")

    def test_flags_only_for_requested_tickers(self):
        from api import news

        with mock.patch.object(news, "_fetch", return_value=NEWS):
            r = news.news_flags(["AAPL", "NVDA", "bad ticker!"])
        self.assertEqual(set(r["flags"]), {"AAPL"})
        self.assertEqual(r["flags"]["AAPL"]["count"], 2)
        self.assertEqual(r["flags"]["AAPL"]["headline"], "Apple beats estimates")

    def test_provider_error_and_kill_switch_answer_empty(self):
        from api import news

        with mock.patch.object(news, "_fetch", side_effect=RuntimeError("down")):
            self.assertFalse(news.stock_news("AAPL")["available"])
            self.assertEqual(news.news_flags(["AAPL"])["flags"], {})
        with mock.patch.dict("os.environ", {"HSF_NEWS_ENABLED": "0"}):
            self.assertFalse(news.stock_news("AAPL")["available"])


# ---- SEC EDGAR ---------------------------------------------------------------------------------------
FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Cook Timothy D</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-10-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.5</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-10-15</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>5000</value></transactionShares></transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


class FilingsTests(unittest.TestCase):
    def setUp(self):
        from api import shared_cache

        shared_cache.reset()
        self.addCleanup(shared_cache.reset)

    def _fake_get(self, url, as_json=True, client=None):
        import datetime as dt

        if url.endswith("company_tickers.json"):
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        if "submissions" in url:
            today = dt.date.today().isoformat()
            return {"filings": {"recent": {
                "form": ["4", "8-K", "10-Q", "4"],
                "filingDate": [today, today, today, "2001-01-01"],
                "accessionNumber": ["0000320193-26-000101", "0000320193-26-000100", "x", "old"],
                "primaryDocument": ["xslF345X05/wk-form4_1.xml", "aapl-8k.htm", "q.htm", "o.xml"],
                "items": ["", "2.02,9.01", "", ""]}}}
        if url.endswith("wk-form4_1.xml"):
            self.assertNotIn("xslF345X05", url)  # the raw XML, not the rendered page
            return FORM4
        raise AssertionError(url)

    def test_parse_form4_keeps_buys_and_sales_only(self):
        from api import filings

        trades = filings.parse_form4(FORM4)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual((t["insider"], t["role"], t["type"], t["value"]),
                         ("Cook Timothy D", "Chief Executive Officer", "Buy", 150500.0))

    def test_stock_filings_reads_insiders_and_8k_items(self):
        from api import filings

        with mock.patch.dict("os.environ", {"HSF_SEC_USER_AGENT": "HSFinest.AI test@example.com"}), \
                mock.patch.object(filings, "_get", side_effect=self._fake_get):
            r = filings.stock_filings("aapl")
        self.assertTrue(r["available"])
        self.assertEqual(r["insider_summary"]["buys"], 1)
        self.assertEqual(r["events"][0]["labels"], ["Earnings results"])
        self.assertIn("/320193/000032019326000101/", r["insider"][0]["url"])

    def test_needs_a_contact_for_edgar(self):
        from api import filings

        with mock.patch.dict("os.environ", {"HSF_SEC_USER_AGENT": "", "SMTP_FROM": ""}):
            self.assertFalse(filings.enabled())
            self.assertFalse(filings.stock_filings("AAPL")["available"])
        with mock.patch.dict("os.environ", {"HSF_SEC_USER_AGENT": "", "SMTP_FROM": "HSF <alerts@hsf.example>"}):
            self.assertEqual(filings.user_agent(), "HSFinest.AI alerts@hsf.example")


class InsiderRuleTests(unittest.TestCase):
    def test_first_pass_sets_a_baseline_then_new_buys_fire_once(self):
        from api.alert_rules import check

        rule = {"rule_type": "INSIDER_BUY", "threshold": 100000.0}
        old = {"insider_known": True, "insider_buy_date": "2026-10-01", "insider_buy_value": 500000.0}
        fires, _, _, st = check(rule, old, None)
        self.assertFalse(fires)  # never fires for buys filed before the rule existed
        new = {**old, "insider_buy_date": "2026-10-15", "insider_buy_value": 150500.0, "insider_buy_who": "X (CEO)"}
        fires, value, _, st2 = check(rule, new, st)
        self.assertTrue(fires)
        self.assertEqual(value, 150500.0)
        self.assertFalse(check(rule, new, st2)[0])
        small = {**new, "insider_buy_date": "2026-10-20", "insider_buy_value": 5000.0}
        self.assertFalse(check(rule, small, st2)[0])
        self.assertIsNone(check(rule, {"insider_known": False}, st2)[3])  # EDGAR down: keep state

    def test_rule_is_pro_only(self):
        from api.alert_rules import rule_types

        basic = {t["type"]: t["available"] for t in rule_types({"can_insider_alerts": False})}
        pro = {t["type"]: t["available"] for t in rule_types({"can_insider_alerts": True})}
        self.assertFalse(basic["INSIDER_BUY"])
        self.assertTrue(pro["INSIDER_BUY"])


# ---- Sentry web stacks -------------------------------------------------------------------------------
class WebStackTests(unittest.TestCase):
    def test_chrome_and_safari_frames_oldest_first_without_queries(self):
        from api.monitoring import parse_js_stack

        stack = ("TypeError: x is undefined\n"
                 "    at Foo (https://hsf-web-v1.onrender.com/_next/static/chunks/a.js?dpl=1:1:2345)\n"
                 "Bar@https://hsf-web-v1.onrender.com/_next/static/chunks/b.js:3:4")
        frames = parse_js_stack(stack)
        self.assertEqual([f["function"] for f in frames], ["Bar", "Foo"])
        self.assertEqual(frames[1]["abs_path"], "https://hsf-web-v1.onrender.com/_next/static/chunks/a.js")
        self.assertEqual((frames[1]["lineno"], frames[1]["colno"]), (1, 2345))

    def test_report_with_release_becomes_a_javascript_exception(self):
        import sentry_sdk

        from api import monitoring
        from ui import monitoring as ui_monitoring

        sent = []
        with mock.patch.object(ui_monitoring, "_initialized", True), \
                mock.patch.object(sentry_sdk, "capture_event", side_effect=lambda e, **k: sent.append(e)):
            monitoring.capture_client_error(
                {"kind": "boundary", "message": "x is undefined", "release": "abc123"},
                "TypeError: x is undefined\n    at Foo (https://h/_next/static/chunks/a.js:1:2)")
        self.assertEqual(sent[0]["release"], "web@abc123")
        self.assertEqual(sent[0]["exception"]["values"][0]["type"], "TypeError")


# ---- API: Turnstile, news/filings routes, chat alerts --------------------------------------------------
@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class IntegrationApiTests(AccountApiTestCase):
    def test_turnstile_off_by_default_and_required_when_on(self):
        self.assertEqual(self.signup().status_code, 201)
        with mock.patch.dict("os.environ", {"TURNSTILE_SECRET_KEY": "secret"}):
            r = self.signup(email="b@example.com", username="b")
            self.assertEqual(r.status_code, 400, r.text)
            self.assertIn("security check", r.json()["detail"])
            ok = mock.Mock(status_code=200, json=lambda: {"success": True})
            with mock.patch("httpx.Client.post", return_value=ok) as post:
                r = self.signup(email="c@example.com", username="c", turnstile_token="tok")
            self.assertEqual(r.status_code, 201, r.text)
            self.assertEqual(post.call_args.kwargs["data"]["response"], "tok")
            bad = mock.Mock(status_code=200, json=lambda: {"success": False, "error-codes": ["invalid-input-response"]})
            with mock.patch("httpx.Client.post", return_value=bad):
                r = self.client.post("/v1/auth/password-reset", json={"email": "c@example.com", "turnstile_token": "x"})
            self.assertEqual(r.status_code, 400)
            with mock.patch("httpx.Client.post", side_effect=ConnectionError("cloudflare down")):
                r = self.client.post("/v1/auth/password-reset", json={"email": "c@example.com", "turnstile_token": "x"})
            self.assertEqual(r.status_code, 202)  # fail open when Cloudflare can't be reached

    def _token(self):
        return self.login().json()["access_token"]

    def test_news_and_filings_routes(self):
        h = self.auth(self._token())
        with mock.patch("api.news.stock_news", return_value={"ticker": "AAPL", "items": [], "available": True}):
            self.assertEqual(self.client.get("/v1/stocks/AAPL/news", headers=h).json()["ticker"], "AAPL")
        with mock.patch("api.news.news_flags", return_value={"flags": {}, "hours": 24}) as f:
            self.assertEqual(self.client.get("/v1/news/flags?symbols=AAPL,MSFT", headers=h).status_code, 200)
        self.assertEqual(f.call_args.args[0], ["AAPL", "MSFT"])
        empty = {"ticker": "AAPL", "available": False, "insider": [], "events": [],
                 "insider_summary": {"buys": 0, "sales": 0, "buy_value": 0, "sale_value": 0, "days": 90}}
        with mock.patch("api.filings.stock_filings", return_value=empty):
            self.assertEqual(self.client.get("/v1/stocks/AAPL/filings", headers=h).status_code, 200)
        self.assertEqual(self.client.get("/v1/stocks/AAPL/news").status_code, 401)

    def test_discord_connect_validates_and_never_echoes_the_url(self):
        from api import chat_delivery

        saved = {}
        hook = "https://discord.com/api/webhooks/123456789012/abcdefghijklmnopqrstuvwxyz_-123"
        with mock.patch.object(chat_delivery, "destinations",
                               side_effect=lambda u: [{"kind": k, "target": v, "label": "Discord channel",
                                                       "last_status": None, "last_at": None} for k, v in saved.items()]), \
                mock.patch.object(chat_delivery, "_save", side_effect=lambda u, k, t, label: saved.update({k: t})), \
                mock.patch.object(chat_delivery, "send_discord", return_value=(True, 204)):
            h = self.auth(self._token())
            r = self.client.put("/v1/me/chat-alerts/discord", headers=h, json={"webhook_url": "https://evil.example/api/webhooks/1/abc" + "x" * 30})
            self.assertEqual(r.status_code, 400)
            r = self.client.put("/v1/me/chat-alerts/discord", headers=h, json={"webhook_url": hook})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertTrue(r.json()["discord"]["connected"])
            self.assertNotIn("webhooks", r.text)
            self.assertEqual(saved["discord"], hook)

    def test_chat_alerts_need_pro(self):
        from api import chat_delivery

        self.accounts["pro@example.com"]["tier"] = "basic"
        with mock.patch.object(chat_delivery, "destinations", return_value=[]):
            h = self.auth(self._token())
            r = self.client.post("/v1/me/chat-alerts/telegram", headers=h)
        self.assertEqual(r.status_code, 403)

    def test_telegram_webhook_needs_the_secret(self):
        from api import chat_delivery

        with mock.patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "s3cret"}), \
                mock.patch.object(chat_delivery, "handle_telegram_update") as handle:
            upd = {"message": {"chat": {"id": 1}, "text": "/start abc"}}
            self.assertEqual(self.client.post("/v1/telegram/webhook", json=upd).status_code, 403)
            r = self.client.post("/v1/telegram/webhook", json=upd, headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"})
            self.assertEqual(r.status_code, 200)
        handle.assert_called_once_with(upd)


class ChatDeliveryTests(unittest.TestCase):
    def test_alert_goes_to_each_chat_and_gone_destinations_are_removed(self):
        from api import chat_delivery

        dests = [{"kind": "telegram", "target": "42"}, {"kind": "discord", "target": "https://discord.com/api/webhooks/1/x"}]
        removed, marked = [], []
        with mock.patch.object(chat_delivery, "destinations", return_value=dests), \
                mock.patch.object(chat_delivery, "send_telegram", return_value=(False, 403)), \
                mock.patch.object(chat_delivery, "send_discord", return_value=(True, 204)) as dc, \
                mock.patch.object(chat_delivery, "remove", side_effect=lambda u, k: removed.append(k)), \
                mock.patch.object(chat_delivery, "_mark", side_effect=lambda u, k, s: marked.append((k, s))), \
                mock.patch("api.store.get_account", return_value={"tier": "pro", "is_active": True}), \
                mock.patch("api.workspace.is_quiet", return_value=False):
            n = chat_delivery.alert_fired("Pro@Example.com", "AAPL crossed 200")
        self.assertEqual(n, 1)
        self.assertEqual(removed, ["telegram"])
        self.assertEqual(marked, [("discord", "accepted")])
        self.assertIn("AAPL crossed 200", dc.call_args.args[1])

    def test_quiet_hours_and_basic_plan_send_nothing(self):
        from api import chat_delivery

        with mock.patch.object(chat_delivery, "destinations", return_value=[{"kind": "telegram", "target": "1"}]), \
                mock.patch.object(chat_delivery, "send_telegram") as tg, \
                mock.patch("api.store.get_account", return_value={"tier": "pro", "is_active": True}), \
                mock.patch("api.workspace.is_quiet", return_value=True):
            self.assertEqual(chat_delivery.alert_fired("u@example.com", "x"), 0)
        tg.assert_not_called()

    def test_telegram_start_links_with_a_fresh_code_only(self):
        from api import chat_delivery

        calls = []

        def run(sql, args, fetch=False):
            calls.append(sql.split()[0])
            if sql.startswith("DELETE FROM hsf_chat_link_codes"):
                return [{"user_id": "pro@example.com"}] if args[0] == "good" else []
            return []

        with mock.patch.object(chat_delivery, "_run", side_effect=run), \
                mock.patch.object(chat_delivery, "_save") as save, \
                mock.patch.object(chat_delivery, "send_telegram") as tg:
            chat_delivery.handle_telegram_update({"message": {"chat": {"id": 7, "first_name": "Howard"}, "text": "/start bad"}})
            save.assert_not_called()
            chat_delivery.handle_telegram_update({"message": {"chat": {"id": 7, "first_name": "Howard"}, "text": "/start good"}})
        save.assert_called_once_with("pro@example.com", "telegram", "7", "Howard")
        self.assertIn("expired", tg.call_args_list[0].args[1])


if __name__ == "__main__":
    unittest.main()
