"""Arrow-safe dataframe sanitizer for st.dataframe.

psycopg returns datetime.date / datetime.datetime objects; pandas keeps them in
object-dtype columns (mixed with None), and pyarrow's convert_column can
SEGFAULT on that shape during Streamlit's Arrow serialization — this was the
app's recurring "crash on first load" (traced via faulthandler to the earnings
panel). Convert such columns to Arrow-native datetime64 before display.
"""
from __future__ import annotations

import datetime as _dt


def arrow_safe(df):
    """Return a copy where object columns of dates/datetimes are datetime64."""
    try:
        import pandas as pd

        if df is None or getattr(df, "empty", True):
            return df
        out = df.copy()
        for col in out.columns:
            if out[col].dtype != object:
                continue
            sample = next((v for v in out[col] if v is not None), None)
            if isinstance(sample, (_dt.date, _dt.datetime)):
                out[col] = pd.to_datetime(out[col], errors="coerce")
        return out
    except Exception:
        return df
