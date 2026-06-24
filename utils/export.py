# ai_scanner/utils/export.py
from __future__ import annotations

import io
import zipfile

import pandas as pd
import streamlit as st


def dataframe(df: pd.DataFrame, *, fill_none: bool=False, width: str="stretch"):
    if fill_none:
        df = df.fillna("")
    st.dataframe(df, width=width)

def download_zip_button(label: str, files: dict[str, bytes|str], filename="bundle.zip"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(name, content)
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="application/zip")