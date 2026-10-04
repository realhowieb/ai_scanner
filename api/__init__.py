"""HSF API (P1-59): one backend for the new web frontend and the iOS/Android app.

Runs as its own service beside the Streamlit app and the billing service, and
reuses the app's Python modules (users table, tiering, scan readers). It never
imports Streamlit.
"""
