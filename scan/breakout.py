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

    We construct a minimal results table from the latest close/volume for each
    symbol in `price_data`, applying basic price filters and limiting to
    `top_n` rows. This ensures scans always return something while we debug the
    full breakout engine.
    """
    # Try to get Streamlit (for in-app debugging), but don't hard-depend on it.
    try:
        import streamlit as st  # type: ignore
    except Exception:
        st = None  # type: ignore[assignment]

    # Debug: how much price data did we actually get?
    if st is not None:
        try:
            st.caption(
                f"📊 Debug: price_data has {len(price_data)} symbols. "
                f"Sample: {list(price_data.keys())[:10]}"
            )
        except Exception:
            pass

    rows: list[dict[str, float | str]] = []

    attempted = len(price_data)
    empty = 0
    no_close = 0
    price_filtered = 0
    added = 0

    for symbol, df_sym in price_data.items():
        if df_sym is None or df_sym.empty:
            empty += 1
            continue

        # Extra safety: normalize column names again on a per-symbol basis.
        try:
            df_sym = df_sym.rename(
                columns=lambda c: str(c).strip().capitalize()
            )
        except Exception:
            # If renaming fails, skip this symbol.
            no_close += 1
            continue

        if "Close" not in df_sym.columns:
            no_close += 1
            continue

        last_row = df_sym.iloc[-1]
        try:
            last_price = float(last_row["Close"])
        except Exception:
            no_close += 1
            continue

        # Basic price range filter
        if not (min_price <= last_price <= max_price):
            price_filtered += 1
            continue

        try:
            volume_val = float(last_row.get("Volume", float("nan")))
        except Exception:
            volume_val = float("nan")

        rows.append(
            {
                "Ticker": symbol,
                "Last": last_price,
                "Volume": volume_val,
            }
        )
        added += 1

    fallback_df = pd.DataFrame(rows)

    # Debug summary of what happened in the fallback.
    if st is not None:
        try:
            st.caption(
                f"🧪 Fallback scan debug: attempted={attempted}, "
                f"added={added}, empty={empty}, "
                f"no_close={no_close}, price_filtered={price_filtered}"
            )
        except Exception:
            pass

    if not fallback_df.empty:
        # Sort by price descending as a simple heuristic and limit rows.
        fallback_df = fallback_df.sort_values("Last", ascending=False).head(top_n)

    return fallback_df
