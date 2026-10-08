"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { useSession } from "@/session/SessionProvider";

const NAV = [
  { href: "/today", label: "Today" },
  { href: "/scanner", label: "Scanner" },
  { href: "/scanner/custom", label: "Custom scan" },
  { href: "/watchlists", label: "Watchlists" },
  { href: "/alerts", label: "Alerts" },
];
export const TICKER_RE = /^[A-Za-z0-9][A-Za-z0-9.-]{0,9}$/;
const CLASSIC = process.env.NEXT_PUBLIC_STREAMLIT_URL || "https://hsfinestai.streamlit.app";

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
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
              const active = n.href === "/scanner" ? path === "/scanner" : path.startsWith(n.href);
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
                <a role="menuitem" href={CLASSIC} className="menu-item">Open classic app</a>
                <button role="menuitem" type="button" className="menu-item" onClick={() => void signOut()}>Sign out</button>
              </div>
            )}
          </div>
        </div>
      </header>
      <main id="main" className="main">{children}</main>
      <footer className="foot">
        <p className="cap">Educational research only, not financial advice.</p>
        <p className="cap">Data: scheduled HSF scans · Alpaca. Setup prices come from the latest scan; SPY, QQQ and the price strip are quotes refreshed every few minutes.</p>
      </footer>
    </div>
  );
}
