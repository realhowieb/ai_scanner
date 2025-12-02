import pandas as pd

def run_breakout_scan(
    price_data,
    spy_df,
    premarket,
    afterhours,
    unusual_volume,
    min_gap,
    min_price,
    max_price,
    top_n,
    diagnostics,
):
    """Simple fallback breakout scan.

    At this stage we don't have a wired `breakout_scanner` engine. Instead we
    construct a minimal results table from the latest close/volume for each
    symbol in `price_data`, applying basic price filters and limiting to
    `top_n` rows. This ensures scans always return something useful while we
    debug the full engine.
    """

    # Debug: show how much price data we actually received.
    try:
        import streamlit as st

        st.caption(
            f"📊 Debug: price_data has {len(price_data)} symbols. "
            f"Sample: {list(price_data.keys())[:10]}"
        )
    except Exception:
        # Ignore if Streamlit isn't active (e.g., during tests or CLI use).
        pass

    rows: list[dict[str, float | str]] = []

    for symbol, df_sym in price_data.items():
        if df_sym is None or df_sym.empty:
            continue

        # Normalize column names (e.g., 'close' -> 'Close', 'volume' -> 'Volume')
        try:
            df_sym = df_sym.rename(columns=lambda c: str(c).strip().capitalize())
        except Exception:
            # If renaming fails for some reason, skip this symbol.
            continue

        if "Close" not in df_sym.columns:
            continue

        last_row = df_sym.iloc[-1]

        try:
            last_price = float(last_row["Close"])
        except Exception:
            continue

        if not (min_price <= last_price <= max_price):
            continue

        volume_val = float(last_row.get("Volume", float("nan")))

        rows.append(
            {
                "Ticker": symbol,
                "Last": last_price,
                "Volume": volume_val,
            }
        )

    fallback_df = pd.DataFrame(rows)

    if not fallback_df.empty:
        # Sort by price descending as a simple heuristic and limit the number of rows.
        fallback_df = fallback_df.sort_values("Last", ascending=False).head(top_n)

    return fallback_df
