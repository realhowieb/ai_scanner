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
    # Main breakout path: use the full breakout scanner if it works.
    try:
        results = breakout_scanner(
            price_data=price_data,
            spy_df=spy_df,
            premarket=premarket,
            afterhours=afterhours,
            unusual_volume=unusual_volume,
            min_gap=min_gap,
            min_price=min_price,
            max_price=max_price,
            top_n=top_n,
            diagnostics=diagnostics,
        )
        if isinstance(results, pd.DataFrame) and not results.empty:
            return results
    except Exception as e:
        # If the legacy breakout engine crashes, surface the error in the UI
        # and fall back to a simpler, price-based ranking so the scan still
        # returns something useful.
        import traceback

        try:
            import streamlit as st

            st.error(f"❌ breakout_scanner crashed: {e}")
            st.code(traceback.format_exc(), language="python")
        except Exception:
            # If Streamlit isn't available, just print the traceback.
            print("breakout_scanner crashed:", e)
            print(traceback.format_exc())

    # Fallback: build a minimal results table from the latest close for each symbol.
    rows: list[dict[str, float | str]] = []
    for symbol, df_sym in price_data.items():
        if df_sym is None or df_sym.empty:
            continue
        if "Close" not in df_sym.columns:
            continue

        last_row = df_sym.iloc[-1]
        last_price = float(last_row["Close"])
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
