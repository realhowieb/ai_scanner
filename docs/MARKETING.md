# Marketing setup: email list, Resend Broadcasts, Buffer

Shipped 2026-11-02. Three pieces in the app, two outside services.

## What's in the app

| Piece | Where | Switch |
|---|---|---|
| Weekly setups email list | Form on `/` and `/weekly`; confirm/unsubscribe page `/subscribe/confirm` | `HSF_MARKETING_LIST_ENABLED=0` hides it (default on) |
| Marketing report | Admin > Marketing | admins only |
| Public weekly recap | `/weekly` (latest) and `/weekly/2026-W44` | always on; a week appears from Saturday 00:00 ET |
| Founding 100 page | `/founding-100` (public, indexable, share image with seats left); every Founding link points here | shows "not open" while `FOUNDING_100_ENABLED` is off |
| Founding 100 emails | Admin > Founding > Founding 100 emails: launch email, then one reminder; preview first, each account gets each once | admins send by hand; never automatic |
| Founding sale alert | Email/chat to the admins when a founder payment completes (tier, number, amount, seats left) | `HSF_OPS_ALERT_EMAIL` picks who; `HSF_OPS_NOTIFY_ENABLED=0` stops all ops messages |

- The list is **double opt-in**: an address only counts after its confirm link is opened. HSF sends only that confirm email; the newsletter itself goes out from Resend.
- Confirmed addresses are copied to Resend automatically when `HSF_RESEND_SEGMENT_ID` is set on hsf-api-v1. Otherwise use Admin > Marketing > Download confirmed (CSV) and import it.
- The recap shows rankings only (days in the top 10, best HSF Score, setup), never returns, so it can't read as a performance claim. Each week is stored the first time someone opens it and never changes.

## Resend Broadcasts (the newsletter)

HSF already sends email through Resend (SMTP). Broadcasts are Resend's newsletter feature, billed separately from transactional email:

- Marketing: free up to 1,000 contacts; Pro Marketing $40/mo for 5,000, $80/mo for 10,000. Sends are not capped.
- Transactional (what HSF uses today): free is capped at 100 emails a day, so the API should be on Pro ($20/mo for 50,000) once digests grow.

One-time setup (about 15 minutes):

1. In Resend, open Audience > Segments and create a segment called "Weekly setups". Copy its ID.
2. Create an API key with **Full access** (a sending-only key can't add contacts). Name it "HSF contacts".
3. On Render hsf-api-v1 set `HSF_RESEND_SEGMENT_ID=<segment id>` and `RESEND_API_KEY=<full-access key>`. Admin > Marketing then shows "Syncing to Resend".
4. In Resend, create a Topic (for example "Weekly setups") so every Broadcast carries Resend's own unsubscribe link. Unsubscribes in Resend stop Broadcasts there; unsubscribes on HSF's page also mark the Resend contact unsubscribed.
5. Use the same verified sending domain as today's alerts; Resend checks SPF/DKIM per domain.

Weekly routine (Saturday or Sunday, about 10 minutes):

1. Open `/weekly` and pick 3 to 5 names worth a sentence.
2. In Resend, Broadcasts > New, audience "Weekly setups". Subject idea: "What HSF ranked this week (Oct 26–30)".
3. Body: two or three short notes, the earnings list, and a button to the recap. Make every link with Admin > Marketing > Campaign link builder (source **Newsletter**), so its visits show up in the report.
4. Send a test to yourself, then schedule it.

Copy rules (finance): describe what the scanner ranked and why; never say a pick "worked", "won", "beat the market" or give returns. Keep the footer line "Research only, not financial advice."

## Buffer (social scheduling)

- Free: 3 channels, 10 queued posts each. Essentials: $6 per channel per month ($5 yearly). 14-day free trial, no card.
- Start free with X, Threads and LinkedIn (or Bluesky). Upgrade only if you post more than 10 a week per channel. Stocktwits isn't in Buffer: post there by hand.
- Every post link should come from the Campaign link builder: pick the page (usually Weekly recap), the network and a campaign like `weekly-w45`. The Campaigns table then shows which posts brought visits, list sign-ups and accounts.

### Automatic weekly drafts (services/social_posts.py)

With `BUFFER_API_KEY` set on hsf-api-v1 (a personal key from publish.buffer.com/settings/api), HSF writes four posts a week and sends each one to every Buffer channel **as a draft**. Nothing publishes until you approve it in Buffer.

| Post | Goes out (ET) | Drafted from |
|---|---|---|
| Earnings this week | Mon 8:30am | Sun 6pm |
| What HSF does (4 rotating posts) | Wed noon | Tue 6pm |
| Founding 100 seats left (skipped when closed) | Fri noon | Thu 6pm |
| Weekly recap top 3 | Sat 10am | Sat 1am |

- Each network gets its own text: Threads and LinkedIn get a link tagged with their utm_source; Instagram gets "link in bio" plus hashtags. Threads posts carry the topic Earnings or Stocks.
- Automatic drafts stop 2 hours before the post time, so a late start never drafts a post you may have already made by hand. **Send now** in Admin still works any time.
- Every post has an image drawn at `/social/<kind>/<week>.png` from the same stored post.
- Text with performance language (returns, wins, "beat", percentages) is refused, never sent.
- A channel that fails is retried on the next pass, up to 3 times. Admin > Marketing > Social drafts shows each post, its preview and status, and has **Send now**.
- `HSF_SOCIAL_DRAFTS_ENABLED=0` pauses it. `HSF_SOCIAL_CHANNELS=instagram,threads` limits it to those networks.
- Set the Instagram profile link to `https://app.hsfinest.com/weekly?utm_source=instagram&utm_medium=social&utm_campaign=bio`.

A simple weekly cadence:

| Day | Post |
|---|---|
| Sat/Sun | "What HSF ranked this week" with 3 tickers and the `/weekly` link |
| Mon | "Biggest earnings this week" from the recap's earnings box |
| Wed | A product view (screenshot of a stock page or Day Trader) with a sign-up link |
| Fri | Founding 100 seats left, with the `/pricing` link (it shows the Founding 100 banner) |

## Reading the report

Admin > Marketing, per source: Visits (browser sessions) → Sign-up clicks → Email list → Accounts → Paid → Founders. Paid plans and founder seats are credited to the buyer's sign-up source. Give a channel 2 to 4 weeks before judging it.
