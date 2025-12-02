def run_breakout_scan(
    tickers: List[str],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool = True,
) -> pd.DataFrame:
    """Public entry point for breakout scans.

    For now, this delegates directly to the legacy `scan.breakout.run_breakout_scan`
    implementation, which has the most stable behaviour in this environment.
    The v2 engine (`run_breakout_scan_v2`) is still available for future use,
    but is not called by default.
    """
    from . import breakout as legacy_breakout

    return legacy_breakout.run_breakout_scan(
        tickers,
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        diagnostics=diagnostics,
    )