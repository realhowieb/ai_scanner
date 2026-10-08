"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { useSession } from "@/session/SessionProvider";

const NAV = [
  { href: "/today", label: "Today" },
  { href: "/brief", label: "Brief" },
  { href: "/scanner", label: "Scanner" },
  { href: "/scanner/custom", label: "Custom scan" },
  { href: "/watchlists", label: "Watchlists" },
  { href: "/alerts", label: "Alerts" },
  { href: "/track-record", label: "Track record" },
];
export const TICKER_RE = /^[A-Za-z0-9][A-Za-z0-9.-]{0,9}$/;
const CLASSIC = process.env.NEXT_PUBLIC_STREAMLIT_URL || "https://hsfinestai.streamlit.app";

/** The in-app page the user came from (path + query), for "Back" links. */
const PreviousPage = createContext<string | null>(null);
export const usePreviousPage = () => useContext(PreviousPage);

const SECTIONS: [string, string][] = [
  ["/today", "Today"], ["/scanner/custom", "Custom scan"], ["/scanner/history", "Scan history"], ["/scanner", "Scanner"],
  ["/watchlists", "Watchlists"], ["/alerts", "Alerts"], ["/track-record", "Track record"], ["/brief", "Market Brief"],
  ["/account", "Account"], ["/stocks", "the previous stock"],
];

/** "Back to Watchlists" for /watchlists?id=3; null when there is no previous page here. */
export function backLabel(prev: string | null): string | null {
  const base = prev?.split("?")[0];
  if (!base) return null;
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

  const search = (e: FormEvent) => {
    e.preventDefault();
    const t = q.trim().toUpperCase();
    if (!TICKER_RE.test(t)) {
      setBad(true);
      return;
    }
    setBad(false);
    setQ("");
    router.push(`/stocks/${encodeURIComponent(t)}`);
  };

  return (
    <div className="shell">
      <a href="#main" className="skip">Skip to content</a>
      <header className="topbar">
        <div className="topbar-in">
          <Link href="/today" className="brand">HSFinest<span>.AI</span></Link>
          <nav aria-label="Main" className="nav">
            {NAV.map((n) => {
              const active = n.href === "/scanner" ? path === "/scanner" || path.startsWith("/scanner/history") : path.startsWith(n.href);
              return (
                <Link key={n.href} href={n.href} className="navlink" aria-current={active ? "page" : undefined}>{n.label}</Link>
              );
            })}
          </nav>
          <form role="search" onSubmit={search} className="search">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></svg>
            <label htmlFor="ticker-search" className="sr-only">Search ticker</label>
            <input id="ticker-search" type="search" placeholder="Search ticker" value={q} maxLength={10} autoComplete="off"
              aria-invalid={bad || undefined} aria-describedby={bad ? "ticker-search-err" : undefined}
              onChange={(e) => { setQ(e.target.value); setBad(false); }} />
            {bad && <span id="ticker-search-err" className="sr-only">Enter a ticker symbol like AAPL</span>}
          </form>
          <div className="account">
            <button type="button" className="btn" aria-expanded={menu} aria-haspopup="true" onClick={() => setMenu((m) => !m)}>
              {me ? `${me.plan_label} · Account` : "Account"}
            </button>
            {menu && (
              <div className="menu" role="menu">
                {me && <p className="cap menu-email">{me.email}</p>}
                <Link role="menuitem" href="/account" className="menu-item" onClick={() => setMenu(false)}>Account &amp; billing</Link>
                <a role="menuitem" href={CLASSIC} className="menu-item">Open classic app</a>
                <button role="menuitem" type="button" className="menu-item" onClick={() => void signOut()}>Sign out</button>
              </div>
            )}
          </div>
        </div>
      </header>
      <main id="main" className="main"><PreviousPage.Provider value={trail.prev}>{children}</PreviousPage.Provider></main>
      <footer className="foot">
        <p className="cap">Educational research only, not financial advice. <Link href="/how-hsf-works">How HSF works</Link></p>
        <p className="cap">Data: scheduled HSF scans · Alpaca. Setup prices come from the latest scan; SPY, QQQ and the price strip are quotes refreshed every few minutes.</p>
      </footer>
    </div>
  );
}
