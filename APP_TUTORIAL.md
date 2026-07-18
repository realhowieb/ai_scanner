# HS Finest AI App Tutorial

This guide walks through the main workflow for using the Streamlit stock scanner:
build a watchlist, run scans, read results, inspect AI signals, and manage alerts.

The app is a research and screening tool. It does not guarantee trades, and it should
not replace your own risk management.

## 1. First Login

1. Open the Streamlit app.
2. Sign in with your account.
3. Confirm your tier in the sidebar:
   - Basic: limited scans and locked premium tools.
   - Pro: exports, scan history, and more result tools.
   - Premium: 3-Step scanner, Early Breakout, AI notes, and advanced signals.
   - Admin: diagnostics and model training tools.

If the app asks for billing or upgrade access, finish the Stripe checkout flow and return
to the app.

## 2. Recommended Daily Workflow

Use this flow when you want a fast morning scan:

1. Check **My Watchlists** near the top of the app.
2. Review the live quote tiles for your saved tickers.
3. Click **Run Watchlist Scan** to scan only your watchlist.
4. Review **Latest scan results**.
5. Open **Watchlist intelligence 2.0** to see what is actionable now and what changed
   since the previous saved scan.
6. Use the chart/details panel to inspect the strongest names.
7. Save or export results if your tier allows it.
8. Create alerts for names you want to monitor.

For a broader scan, use the manual scan controls or the **EZ 3-Step AI Scanner**.

## 3. My Watchlists

The **My Watchlists** section helps you keep a short list of tickers you care about.

Common actions:

- Use the watchlist dropdown to choose the active list.
- Click **+ New** to create a new watchlist.
- Type a ticker in **Add ticker** and click **Add**.
- Click **Run Watchlist Scan** to scan only the active list.
- Click **View as table** to view quote data for the list.
- Click **CSV** to export the ticker list.
- Open **Manage** to remove symbols, bulk edit the list, clear it, or delete the watchlist.

## 4. Watchlist Intelligence 2.0

After a scan finishes, open **Watchlist intelligence 2.0** under your watchlist tiles.

The **Now** table shows each watchlist ticker matched against the latest scan:

- **Active breakout**: the breakout flag or AI signal is already strong.
- **Heating up**: pre-breakout probability is elevated.
- **Strong setup**: breakout score is elevated.
- **Cooling down**: model signals are quiet.
- **Watching**: no strong signal yet.
- **Not in latest scan**: the ticker did not pass the latest scan filters.

The **Since previous saved scan** table compares the latest scan to the previous saved run:

- **Heating up**: model score, AI confidence, or breakout score improved meaningfully.
- **Cooling down**: the setup weakened meaningfully.
- **New in latest scan**: the ticker newly appeared in the latest scan.
- **Dropped out**: the ticker was present before but not in the latest scan.

Stable rows are hidden by default. Use **Show stable rows** if you want to audit unchanged names.

## 5. Day Trader Panel

The **Day Trader** panel shows live mover data such as gappers, VWAP, and relative volume.

Use it when you want a faster real-time view before running a full scan. If a provider is
rate-limited or unavailable, this panel may show fewer rows or no rows temporarily.

## 6. Manual Scan

Use the manual scan controls when you want more control over the scan inputs.

Typical settings:

- **Market universe**: choose S&P 500, NASDAQ, or Combo.
- **Profile**: choose aggressive, regular, or conservative.
- **Top N Results**: controls how many names are returned.
- **Min price / Max price**: filters stocks by price.
- **Min gap %**: filters for gap setups.
- **Unusual volume**: filters for volume pressure.
- **Premarket / after-hours**: uses extended-hours pricing when enabled.

After choosing settings, run the scan and review the **Latest scan results** tab.

## 7. EZ 3-Step AI Scanner

Premium users can use the guided scanner:

1. **Select Market Universe**
   Choose S&P 500, NASDAQ, or Combo.
2. **Select Strategy**
   Choose the setup type, such as gap up, gap down, most active, unusual volume,
   momentum, or breakout-only.
3. **Profile, Run Scan & View Results**
   Pick aggressive, regular, or conservative, then run the scan.

The 3-Step scanner automatically adds pre-breakout and AI scoring when the model is available.

## 8. Reading Scan Results

The results table is designed for quick scanning.

Important columns:

- **Ticker**: stock symbol.
- **10-day / Spark10D**: mini price trend.
- **Why**: short explanation of the setup.
- **Score / BreakoutScore**: overall breakout setup strength.
- **Last**: latest price used by the scan.
- **Volume**: latest volume.
- **Day % / PctChange**: current price change.
- **Gap % / GapPct**: gap from the prior close.
- **Trend 20d / Trend 10d**: recent trend strength.
- **RVOL / VolRel20**: relative volume.
- **$Vol 20d / DollarVol20**: dollar volume liquidity.
- **PatternTag**: setup label such as Base/Neutral, GapDown, NearHigh.
- **RS vs SPY**: relative strength versus SPY.
- **PreBreakout**: model probability that a ticker has a pre-breakout pattern.

Click a row to drive the chart and details panel. Use the CSV export button if your tier
allows exports.

## 9. Charts And Details

After a scan returns rows:

1. Open **Charts**.
2. Click a table row to change the chart ticker.
3. Use **Change chart ticker** if you want to manually choose another ticker.
4. Open the ticker details panel to see key stats, row fields, and trade planning tools.

Basic users see one auto-selected ticker details panel. Pro and Premium users can select
specific rows and export results.

## 10. Early Breakout Candidates

The **Early Breakout Candidates** tab uses the trained model to rank stocks before a
confirmed breakout.

How to use it:

1. Run a scan first.
2. Open **Early Breakout Candidates**.
3. Confirm a model is loaded from the database.
4. Adjust the minimum pre-breakout probability threshold.
5. Review the highest-ranked candidates.

If the tab says no model is loaded, an admin needs to train or upload the model. Once saved
to the database, users do not need to retrain it after reboot.

## 11. AI Confidence

AI Confidence is generated by the trained XGBoost model when the model and required feature
columns are available.

The model uses these features:

- `Trend10D%`
- `Trend20D%`
- `VolRel20`
- `DollarVol20`
- `BreakoutScore`
- `GapPct`

If the model or features are missing, the app continues without AI scores instead of crashing.

## 12. Track Record

The **Track Record** tab helps review whether historical scan picks performed well after
the scan date.

Use it to answer:

- Did high-scoring picks work better than lower-scoring picks?
- Did pre-breakout candidates follow through?
- How did picks perform versus a benchmark?

This is best used over time after you have saved enough scans.

## 13. Scan History

The **Scan History** tab shows saved scan runs from the database.

Use it to:

- Reopen past scans.
- Compare older results with current results.
- Review row count and scan duration.
- Confirm that scans are being saved correctly.

If scan history is empty, run a scan and wait for it to save.

## 14. AI Notes, Summary, And Chat

Premium users may see AI tools after scan results:

- **AI Summary**: summarizes the current scan.
- **What changed**: compares current results to previous saved results.
- **AI Chat**: asks questions about the current scan.
- **Ticker analysis**: explains a selected ticker setup.

Use these tools to speed up review, then verify the chart and metrics yourself.

## 15. Alerts

Open **Alerts** to create and manage notifications.

Common alert types:

- Breakout score threshold.
- Watchlist membership.
- Price above or below a level.
- Live percent move.
- Relative volume.

After creating alerts, check the alert fired count near the top of the app. Alert details
appear in the Alerts section.

## 16. Admin And Model Training

Admin users can train or refresh the pre-breakout / AI confidence model.

Typical admin flow:

1. Confirm the app can connect to the production database.
2. Open the model status panel.
3. Click **Train / Refresh model from DB history**.
4. Confirm the model reports:
   - Model loaded: yes
   - Source: database
   - AUC
   - trained_at
   - feature count
5. Rerun a scan and confirm PreBreakout / AI Confidence appears where expected.

AUC may move up or down as more realistic scan history is added. A lower AUC is not
automatically bad if it means the model is learning from harder, more recent examples.

## 17. Troubleshooting

### The app shows zero scan results

Try:

- Lower **Min Gap %**.
- Widen the price range.
- Turn off unusual volume.
- Use a less strict profile.
- Try another market universe.

### A ticker says "Not in latest scan"

That ticker is in your watchlist but did not pass the latest scan filters. Run a broader
scan or loosen filters.

### A provider is rate-limited

Market data providers can temporarily rate-limit requests. Wait a few minutes, use cached
mode when available, or reduce the number of symbols scanned.

### Earnings shows blank or TBA

The upstream earnings provider may not have a confirmed date/time. This is normal for some
symbols.

### The model is unavailable

The app should continue without model scores. Admin can train or upload the model to the
database so it loads on startup.

## 18. Best Practices

- Start with watchlists for speed.
- Use broader scans to discover new names.
- Treat PreBreakout and AI Confidence as ranking signals, not trade commands.
- Confirm every candidate on the chart.
- Avoid chasing low-liquidity tickers.
- Use alerts for names you want to monitor, not every ticker in a scan.
- Review Track Record weekly to see what is actually working.

