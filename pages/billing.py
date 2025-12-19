def _redirect_same_tab(url: str) -> None:
    u = (url or "").strip()
    if not u:
        return

    st.success("Stripe Checkout is ready ✅")
    st.caption("Open checkout in the same tab to keep your session. If your browser blocks it, use the fallback link.")

    # Same-tab open (most reliable with Streamlit)
    st.markdown(
        f"""
        <a href={u!r} target="_self" style="text-decoration:none;">
          <button style="width:100%; padding:0.6rem 1rem; border-radius:0.5rem; border:1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.06); color: inherit; font-size: 1rem; cursor: pointer;">
            💳 Open Stripe Checkout (same tab)
          </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

    # Fallback (if the browser forces a new tab anyway)
    st.link_button("Open Stripe Checkout (fallback)", u, use_container_width=True)

    st.stop()
