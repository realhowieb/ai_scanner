# HSF app guide

Use HSF at [app.hsfinest.com](https://app.hsfinest.com). This guide follows the
current web app. On a phone, some navigation items are under **More**.

## Start with today's market

Sign in, then open **Today** for the latest ranked stocks and market context.
Check the scan time and market session before comparing readings. **Brief** adds
market context; **Track record** shows recorded outcomes. Screening results are
research candidates, and a high score does not establish a profitable trade.

## Run your first scan

1. Open **Scanner** to review the latest scheduled results.
2. Choose **Custom scan** to set your own universe and filters. Available tools
   and row limits depend on the plan shown in your account.
3. Under **Advanced filters**, select one **RSI setup (experimental)**:
   **Rebound candidates**, **Momentum recovery**, or **Uptrend pullback**.
   Read the explanation beneath the selector before choosing.
4. Add optional minimum/maximum daily RSI limits if needed. Those limits apply
   in addition to the setup; leaving them blank adds no RSI bound.
5. Optionally select **MACD confirmation**. Above its signal line and a fresh
   bullish crossover are different checks; MACD can still be below zero.
6. Select **Start scan** and wait for the results. Describing a scan and choosing
   **Fill in settings** only fills the controls; it does not run a scan.

If no stocks match, review your universe and combined filters. An empty result
is valid; it does not mean a setup must exist. Custom results use the saved scan
snapshot, so they can differ from a chart fetched later.

## Understand a result

Open a ticker to view its stock page. Use the section links to inspect **Chart**,
**RSI / MACD**, **HSF Score**, **Signals**, **Risks**, **History**, and **Trade plan**.
Availability depends on your plan and the data for that ticker.

In **RSI / MACD**, begin with the selected scan's setup and required checks.
Expand **View rule-by-rule checks** to see each reading. Other indicators are
optional context unless you required them in that scan. A rebound passing while
MACD is unconfirmed is consistent when MACD was not required. Separate chart
checks, when available, can use different history from the saved scan.

**HSF Score**, **PreBreakout**, and RSI/MACD screening checks answer different
questions. They are not interchangeable probabilities. Model evidence and
training status are available to administrators under **Admin → Models**;
passing data readiness does not automatically train or replace a model.

## Save a candidate and return to it

Use the stock's save/watchlist control, then open **Watchlists** to organize saved
stocks. Return to a saved stock to review new readings and risks. Saving a stock
does not place an order or automatically subscribe you to every alert.

Open **Alerts** to review or create the alerts your plan supports. Confirm the
condition and delivery settings. **Journal** lets you record your reasoning;
**Paper trading** is a separate simulation workflow, not a live brokerage order.

## Manage your account

Open **Account** for your plan, preferences, and available upgrade or subscription
management actions. After Stripe returns you to HSF, allow plan confirmation to
finish. A checkout return alone does not prove payment or grant access; the
server verifies subscription state. If access has not updated, refresh the plan
or return to Account rather than paying again.

For questions or a reproducible issue, use the app's help/contact link and include
the page, scan time and steps taken. Never share passwords, session tokens or
payment details.
