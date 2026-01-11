import streamlit as st

st.set_page_config(page_title="Payment canceled", layout="centered")

st.title("⚠️ Payment canceled")
st.write("No worries — you weren’t charged. You can close this tab and return to the app.")

col1, col2 = st.columns(2)
with col1:
    st.link_button("⬅️ Back to Billing", "/billing")
with col2:
    st.link_button("🏠 Open Scanner", "/")