"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode, RefObject } from "react";

import { useSession } from "@/session/SessionProvider";

/** `top`: in the desktop bar (else under More); `mid`: moves under More on narrow desktops.
 * `tab`: in the phone tab bar (else under its More). */
const NAV = [
  { href: "/today", label: "Today", top: true, tab: true },
  { href: "/brief", label: "Brief", top: true, mid: true, tab: false },
  { href: "/scanner", label: "Scanner", top: true, tab: true },
  { href: "/stocks", label: "Stocks", top: true, tab: false },
  { href: "/day-trader", label: "Day Trader", top: true, mid: true, tab: false },
  { href: "/watchlists", label: "Watchlists", top: true, tab: true },
  { href: "/alerts", label: "Alerts", top: true, tab: true },
  { href: "/track-record", label: "Track record", top: false, tab: false },
];

const isActive = (path: string, href: string) => path === href || path.startsWith(`${href}/`);

/** Closes an open menu on Escape (focus back to its button), a click outside it, or a route change. */
export function useDismiss(open: boolean, close: () => void, box: RefObject<HTMLElement | null>, path: string) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      close();
      box.current?.querySelector<HTMLElement>("[aria-haspopup]")?.focus();
    };
    const onDown = (e: PointerEvent) => { if (box.current && !box.current.contains(e.target as Node)) close(); };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => { document.removeEventListener("keydown", onKey); document.removeEventListener("pointerdown", onDown); };
  }, [open, close, box]);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- close on navigation only
  useEffect(() => close(), [path]);
}

type NavItem = { href: string; label: string; mid?: boolean };

function MoreMenu({ items, path, id, className }: { items: NavItem[]; path: string; id: string; className: string }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  useDismiss(open, close, box, path);
  // A `mid` item is in the bar on wide screens, so it only marks More where it has moved under it.
  const active = items.some((n) => !n.mid && isActive(path, n.href));
  const activeMid = items.some((n) => n.mid && isActive(path, n.href));
  return (
    <div className={`more ${className}`} ref={box}>
      <button type="button" className="navlink more-btn" aria-expanded={open} aria-haspopup="true" aria-controls={id}
        data-active={active || undefined} data-active-mid={activeMid || undefined} onClick={() => setOpen((o) => !o)}>
        More<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true"><path d="M6 9l6 6 6-6" /></svg>
      </button>
      {open && (
        <div className="menu more-menu" role="menu" id={id}>
          {items.map((n) => (
            <Link key={n.href} role="menuitem" href={n.href} className={`menu-item${n.mid ? " more-mid" : ""}`} aria-current={isActive(path, n.href) ? "page" : undefined}
              onClick={() => setOpen(false)}>{n.label}</Link>
          ))}
        </div>
      )}
    </div>
  );
}
export const TICKER_RE = /^[A-Za-z0-9][A-Za-z0-9.-]{0,9}$/;
const CLASSIC = process.env.NEXT_PUBLIC_STREAMLIT_URL || "https://hsfinestai.streamlit.app";

/** The in-app page the user came from (path + query), for "Back" links. */
const PreviousPage = createContext<string | null>(null);
export const usePreviousPage = () => useContext(PreviousPage);

const SECTIONS: [string, string][] = [
  ["/today", "Today"], ["/scanner/custom", "Custom scan"], ["/scanner/history", "Scan history"], ["/scanner", "Scanner"],
  ["/watchlists", "Watchlists"], ["/alerts", "Alerts"], ["/track-record", "Track record"], ["/brief", "Market Brief"],
  ["/account", "Account"], ["/journal", "Journal"], ["/day-trader", "Day Trader"], ["/stocks", "Stock Intelligence"], ["/paper", "Paper trading"],
];

/** "Back to Watchlists" for /watchlists?id=3; null when there is no previous page here. */
export function backLabel(prev: string | null): string | null {
  const base = prev?.split("?")[0];
  if (!base) return null;
  if (base.startsWith("/stocks/")) return "Back to the previous stock";
  const hit = SECTIONS.find(([p]) => base === p || base.startsWith(`${p}/`));
  return hit ? `Back to ${hit[1]}` : "Back";
}

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const [trail, setTrail] = useState<{ prev: string | null; cur: string | null }>({ prev: null, cur: null });
  useEffect(() => {
    // Query strings are read here, not with useSearchParams, so the shell needs no Suspense boundary.
    const here = path + (typeof window === "undefined" ? "" : window.location.search);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- records the route change the router just made
    setTrail((t) => (t.cur && t.cur.split("?")[0] === path ? { ...t, cur: here } : { prev: t.cur, cur: here }));
  }, [path]);
  const { me, signOut } = useSession();
  const [q, setQ] = useState("");
  const [bad, setBad] = useState(false);
  const [menu, setMenu] = useState(false);
  const accountBox = useRef<HTMLDivElement>(null);
  const closeMenu = useCallback(() => setMenu(false), []);
  useDismiss(menu, closeMenu, accountBox, path);

  const search = (e: FormEvent) => {
    e.preventDefault();
    const t = q.trim().toUpperCase();
    if (!TICKER_RE.test(t)) {
      setBad(true);
      return;
    }
    setBad(false);
    setQ("");
    setMenu(false);
    (document.activeElement as HTMLElement | null)?.blur();
    router.push(`/stocks/${encodeURIComponent(t)}`);
  };

  return (
    <div className="shell">
      <a href="#main" className="skip">Skip to content</a>
      <header className="topbar">
        <div className="topbar-in">
          <Link href="/today" className="brand">HSFinest<span>.AI</span></Link>
          <nav aria-label="Main" className="nav">
            {NAV.filter((n) => n.top).map((n) => (
              <Link key={n.href} href={n.href} className={`navlink${n.mid ? " nav-mid" : ""}`} aria-current={isActive(path, n.href) ? "page" : undefined}>{n.label}</Link>
            ))}
            <MoreMenu items={[...NAV.filter((n) => n.mid), ...NAV.filter((n) => !n.top)]} path={path} id="more-desk" className="" />
          </nav>
          <form role="search" onSubmit={search} className="search">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></svg>
            <label htmlFor="ticker-search" className="sr-only">Search ticker</label>
            <input id="ticker-search" type="search" placeholder="Search ticker" value={q} maxLength={10} autoComplete="off"
              aria-invalid={bad || undefined} aria-describedby={bad ? "ticker-search-err" : undefined}
              onChange={(e) => { setQ(e.target.value); setBad(false); }} />
            {bad && <span id="ticker-search-err" className="sr-only">Enter a ticker symbol like AAPL</span>}
          </form>
          <div className="account" ref={accountBox}>
            <button type="button" className="btn" aria-expanded={menu} aria-haspopup="true"
              aria-label={me ? `${me.plan_label} · Account` : undefined} onClick={() => setMenu((m) => !m)}>
              {me ? <>{me.plan_label}<span className="hide-narrow">&nbsp;· Account</span></> : "Account"}
            </button>
            {menu && (
              <div className="menu" role="menu">
                {me && <p className="cap menu-email">{me.email}</p>}
                <Link role="menuitem" href="/account" className="menu-item" onClick={() => setMenu(false)}>Account &amp; billing</Link>
                <Link role="menuitem" href="/journal" className="menu-item" onClick={() => setMenu(false)}>Journal</Link>
                <Link role="menuitem" href="/paper" className="menu-item" onClick={() => setMenu(false)}>Paper trading</Link>
                <Link role="menuitem" href="/pricing" className="menu-item" onClick={() => setMenu(false)}>Plans and pricing</Link>
                <a role="menuitem" href={CLASSIC} className="menu-item">Classic app</a>
                <button role="menuitem" type="button" className="menu-item" onClick={() => void signOut()}>Sign out</button>
              </div>
            )}
          </div>
        </div>
      </header>
      <nav aria-label="Sections" className="tabbar">
        {NAV.filter((n) => n.tab).map((n) => (
          <Link key={n.href} href={n.href} className="tab" aria-current={isActive(path, n.href) ? "page" : undefined}>{n.label}</Link>
        ))}
        <MoreMenu items={NAV.filter((n) => !n.tab).map(({ href, label }) => ({ href, label }))} path={path} id="more-tab" className="tab-more" />
      </nav>
      <main id="main" className="main"><PreviousPage.Provider value={trail.prev}>{children}</PreviousPage.Provider></main>
      <footer className="foot">
        <p className="cap">HSF Score is an opportunity ranking, not a probability of profit. Educational research only, not financial advice. <Link href="/how-hsf-works">How HSF works</Link></p>
        <p className="cap">Data: scheduled HSF scans · Alpaca. Setup prices come from the latest scan; SPY, QQQ and the price strip are quotes refreshed every few minutes.</p>
      </footer>
    </div>
  );
}
