import streamlit as st
import pandas as pd
import time

from ai_scanner.runs import save_run, list_runs, load_run_results

def main():
    # ... existing main function code ...

    def do_scan():
        # ... existing do_scan code ...
        df = safe_call(
            cached_real_scan,
            tuple(tickers),
            premarket=premarket,
            afterhours=afterhours,
            unusual_volume=unusual_vol,
            min_gap=min_gap,
            min_price=min_price,
            max_price=max_price,
            top_n=top_n,
            diagnostics=diagnostics,
            label="cached_real_scan",
        )

        # Apply Top N cap here to avoid doing last-price overrides on hundreds of rows.
        if df is not None and not df.empty:
            df = df.head(top_n).reset_index(drop=True)
            df = _override_last_prices(df)

        st.caption(f"✅ {label}: {len(df)} results returned from scan.")
        dt = time.time() - t0
        st.session_state.results_df = df
        banner(f"✅ {label} scan complete in {dt:.1f}s. Returned {len(df)} rows.", "success")

        # Persist this scan to the runs DB (Neon / SQLAlchemy backend)
        try:
            results_json = df.to_json(orient="records") if df is not None else "[]"
            run_name = f"{label} | {len(df)} results | {dt:.1f}s"
            save_run(run_name, results_json)
        except Exception:
            # Never fail the UI just because DB logging failed
            pass

    # ... rest of main function code ...

    # --- Scan History (DB-backed via ai_scanner.runs) ---
    with st.expander("📜 Scan History", expanded=False):
        runs_list = []
        try:
            runs_list = list_runs()
        except Exception as e:
            st.caption("History unavailable (DB error).")

        if runs_list:
            # Build nice labels like "#12 — SP500 scan | 45 results | 23.1s — 2025-11-25 08:03"
            options = []
            for r in runs_list:
                ts = r.get("timestamp")
                ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                label_str = f"#{r['id']} — {r['name']} — {ts_str}"
                options.append((label_str, r["id"]))

            labels = [lbl for (lbl, _rid) in options]
            selected_label = st.selectbox("Select a past scan to load:", labels, index=0)
            selected_id = None
            for lbl, _rid in options:
                if lbl == selected_label:
                    selected_id = _rid
                    break

            col_hist1, col_hist2 = st.columns([1, 1])
            with col_hist1:
                if st.button("Load Selected Scan") and selected_id is not None:
                    try:
                        payload = load_run_results(int(selected_id))
                        hist_df = pd.read_json(payload)
                        st.session_state.results_df = hist_df
                        st.success(f"Loaded scan #{selected_id} from history with {len(hist_df)} rows.")
                    except Exception as e:
                        st.error(f"Failed to load scan #{selected_id}: {e}")
            with col_hist2:
                st.caption("History is stored in your configured DB_URL (Neon/Postgres) or local scanner.sqlite if unset.")
        else:
            st.caption("No past scans saved yet.")
