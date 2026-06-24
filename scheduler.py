# scheduler.py
from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import streamlit as st


@dataclass
class RunFns:
    run_premarket: Callable[[], int]
    run_postmarket: Callable[[], int]
    run_sp500: Callable[[], int]


# Process-level singleton so concurrent Streamlit sessions share one scheduler
# instead of each spawning their own background thread.
_process_scheduler: BackgroundScheduler | None = None
_process_scheduler_lock = threading.Lock()
_process_scheduler_started: bool = False


def _warn_if_multi_worker() -> None:
    """Emit a one-time warning when the app appears to run with multiple workers."""
    import os
    workers = int(os.environ.get("WEB_CONCURRENCY", os.environ.get("STREAMLIT_SERVER_WORKERS", "1")))
    if workers > 1 and not st.session_state.get("_sched_multiworker_warned"):
        st.warning(
            f"⚠️ Scheduler: detected {workers} workers. "
            "APScheduler's BackgroundScheduler is a single-process singleton — "
            "jobs will only fire in ONE worker process. "
            "For reliable multi-worker scheduling, use an external queue (Celery/RQ) or "
            "a dedicated scheduler process."
        )
        st.session_state["_sched_multiworker_warned"] = True


def get_scheduler(tz_str: str = "America/New_York") -> BackgroundScheduler:
    """Return the process-level BackgroundScheduler (shared across all Streamlit sessions)."""
    global _process_scheduler, _process_scheduler_started
    with _process_scheduler_lock:
        if _process_scheduler is None:
            tz = pytz.timezone(tz_str)
            _process_scheduler = BackgroundScheduler(timezone=tz)
            _process_scheduler_started = False
    # Mirror into session_state so existing callers can still read it from there.
    st.session_state["scheduler"] = _process_scheduler
    st.session_state["scheduler_started"] = _process_scheduler_started
    return _process_scheduler


def _purge_old_login_attempts() -> None:
    """Delete login_attempts rows older than 24 hours. Runs nightly."""
    try:
        from db.engine import get_neon_conn
        conn = get_neon_conn()
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute("DELETE FROM login_attempts WHERE attempted_at < NOW() - INTERVAL '24 hours'")
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        print(f"[scheduler] purged {deleted} stale login_attempts rows")
    except Exception as e:
        print(f"[scheduler] login_attempts purge failed: {e}")


def clear_jobs(scheduler: BackgroundScheduler) -> None:
    for job in scheduler.get_jobs():
        scheduler.remove_job(job.id)


def add_default_jobs(
    scheduler: BackgroundScheduler,
    fns: RunFns,
    tz_str: str = "America/New_York",
    pre_time: Optional[Tuple[int, int]] = (8, 30),     # 08:30 ET
    intraday_time: Optional[Tuple[int, int]] = (9, 45),# 09:45 ET
    post_time: Optional[Tuple[int, int]] = (16, 10),   # 16:10 ET
) -> None:
    """Define/replace the standard jobs."""
    clear_jobs(scheduler)

    if pre_time is not None:
        h, m = pre_time
        scheduler.add_job(
            fns.run_premarket,
            trigger=CronTrigger(day_of_week="mon-fri", hour=h, minute=m, timezone=tz_str),
            id="premarket_job",
            replace_existing=True,
        )

    if intraday_time is not None:
        h, m = intraday_time
        scheduler.add_job(
            fns.run_sp500,
            trigger=CronTrigger(day_of_week="mon-fri", hour=h, minute=m, timezone=tz_str),
            id="sp500_intraday_job",
            replace_existing=True,
        )

    if post_time is not None:
        h, m = post_time
        scheduler.add_job(
            fns.run_postmarket,
            trigger=CronTrigger(day_of_week="mon-fri", hour=h, minute=m, timezone=tz_str),
            id="postmarket_job",
            replace_existing=True,
        )

    # Nightly maintenance: purge stale login_attempts rows
    scheduler.add_job(
        _purge_old_login_attempts,
        trigger=CronTrigger(hour=2, minute=0, timezone=tz_str),
        id="purge_login_attempts",
        replace_existing=True,
    )


def start(scheduler: BackgroundScheduler) -> None:
    global _process_scheduler_started
    with _process_scheduler_lock:
        if not _process_scheduler_started:
            scheduler.start()
            _process_scheduler_started = True
            st.session_state["scheduler_started"] = True
    _warn_if_multi_worker()


def pause(scheduler: BackgroundScheduler) -> None:
    try:
        scheduler.pause()
    except Exception:
        pass


def render_jobs_sidebar(st_mod, scheduler: BackgroundScheduler) -> None:
    """Tiny helper to show job list in the sidebar."""
    jobs = scheduler.get_jobs()
    if jobs:
        for j in jobs:
            st_mod.sidebar.caption(f"• {j.id} → next: {j.next_run_time}")
    else:
        st_mod.sidebar.caption("No jobs scheduled.")


def render_sidebar_controls(
    fns: RunFns,
    tz_str: str = "America/New_York",
    pre_time: Optional[Tuple[int, int]] = (8, 30),
    intraday_time: Optional[Tuple[int, int]] = (9, 45),
    post_time: Optional[Tuple[int, int]] = (16, 10),
) -> None:
    """One-call helper used by main.py to wire the sidebar UI."""
    st.sidebar.markdown("### Scheduler")
    enable_sched = st.sidebar.checkbox("Enable automatic scheduled runs", value=True, key="sched_enabled")

    scheduler = get_scheduler(tz_str=tz_str)

    if enable_sched:
        add_default_jobs(
            scheduler,
            fns=fns,
            tz_str=tz_str,
            pre_time=pre_time,
            intraday_time=intraday_time,
            post_time=post_time,
        )
        start(scheduler)
        render_jobs_sidebar(st, scheduler)

        # Optional “Run now” test buttons
        st.sidebar.markdown("**Run now (manual)**")
        c1, c2, c3 = st.sidebar.columns(3)
        if c1.button("Pre", key="sched_run_pre_now"):
            rid = fns.run_premarket()
            st.sidebar.success(f"Pre-market saved as run #{rid}")
        if c2.button("S&P", key="sched_run_sp_now"):
            rid = fns.run_sp500()
            st.sidebar.success(f"S&P 500 saved as run #{rid}")
        if c3.button("Post", key="sched_run_post_now"):
            rid = fns.run_postmarket()
            st.sidebar.success(f"Post-market saved as run #{rid}")
    else:
        pause(scheduler)