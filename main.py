def load_universe(file_path="universe.txt", min_price=1.0, max_price=1500.0, exchange="all"):
    from pathlib import Path
    p = Path(file_path)
    if p.exists() and p.stat().st_size > 0:
        tickers = p.read_text().splitlines()
        st.info(f"Loaded universe from {file_path} with {len(tickers)} tickers")
        return tickers

    exchanges = ["nasdaq", "nyse", "amex"] if exchange == "all" else [exchange.lower()]
    all_tickers = []

    for exch in exchanges:
        try:
            if exch == "nasdaq":
                tickers = si.tickers_nasdaq()
            elif exch == "nyse":
                tickers = si.tickers_nyse()
            elif exch == "amex":
                tickers = si.tickers_amex()
            else:
                tickers = []
            all_tickers.extend(tickers)
        except Exception as e:
            st.warning(f"Failed to fetch tickers for {exch.upper()}: {e}")
            continue

    all_tickers = sorted(set(all_tickers))
    valid_tickers = []
    skipped_tickers = []

    batch_size = 100
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="2d", interval="1d", progress=False, threads=True, auto_adjust=False)
        except Exception:
            skipped_tickers.extend(batch)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"].iloc[-1]
            for t in batch:
                try:
                    price = closes[t] if t in closes else np.nan
                    if pd.isna(price):
                        skipped_tickers.append(t)
                    elif min_price <= price <= max_price:
                        valid_tickers.append(t)
                    else:
                        skipped_tickers.append(t)
                except Exception:
                    skipped_tickers.append(t)
        else:
            # Single ticker case
            try:
                price = data["Close"].iloc[-1]
                if min_price <= price <= max_price:
                    valid_tickers.append(batch[0])
                else:
                    skipped_tickers.append(batch[0])
            except Exception:
                skipped_tickers.append(batch[0])

    if skipped_tickers:
        st.warning(f"Skipped {len(skipped_tickers)} tickers due to missing data or delisted: "
                   f"{', '.join(skipped_tickers[:20])}{'...' if len(skipped_tickers) > 20 else ''}")

    valid_tickers = [t.replace(".", "-") for t in valid_tickers]
    p.write_text("\n".join(valid_tickers))
    st.info(f"Universe loaded: {len(valid_tickers)} tickers from {', '.join(exchanges).upper()} "
            f"between ${min_price} and ${max_price}")
    return valid_tickers