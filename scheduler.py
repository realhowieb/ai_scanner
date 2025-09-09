# scheduler.py
from __future__ import annotations

from typing import Callable, Optional, Tuple
from dataclasses import dataclass

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import streamlit as st


@dataclass
class RunFns:
    """Callables supplied by main.py so we avoid circular imports."""
    run_premarket: Callable[[], int]
    run_postmarket: Callable[[], int]
    run_sp500: Callable[[], int]


def get_scheduler(tz_str: str = "America/New_York") -> BackgroundScheduler:
    """Return a single BackgroundScheduler stored in session_state."""
    if "scheduler" not in st.session_state:
        tz = pytz.timezone(tz_str)
        st.session_state["scheduler"] = BackgroundScheduler(timezone=tz)
        st.session_state["scheduler_started"] = False
    return st.session_state["scheduler"]


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


def start(scheduler: BackgroundScheduler) -> None:
    if not st.session_state.get("scheduler_started", False):
        scheduler.start()
        st.session_state["scheduler_started"] = True


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