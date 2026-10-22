"use client";

import Link from "@/components/Link";
import { usePathname } from "next/navigation";
import { DropdownMenu } from "radix-ui";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

import { StockSearch } from "@/components/StockSearch";
import { InstallHint } from "@/components/InstallHint";
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

/** Keeps a controlled menu closed after the route changes. */
function useCloseOnNavigate(close: () => void, path: string) {
  // eslint-disable-next-line react-hooks/exhaustive-deps -- close on navigation only
  useEffect(() => close(), [path]);
}

type NavItem = { href: string; label: string; mid?: boolean };

// Menus use Radix DropdownMenu: arrow keys, Home/End and type-ahead move between items,
// Escape closes and returns focus to the button. Rendered in place (no portal) so the
// menu stays inside its nav landmark; modal={false} keeps the page usable while it is open.
function MoreMenu({ items, path, id, className, side = "bottom" }: { items: NavItem[]; path: string; id: string; className: string; side?: "top" | "bottom" }) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  useCloseOnNavigate(close, path);
  // A `mid` item is in the bar on wide screens, so it only marks More where it has moved under it.
  const active = items.some((n) => !n.mid && isActive(path, n.href));
  const activeMid = items.some((n) => n.mid && isActive(path, n.href));
  return (
    <div className={`more ${className}`}>
      <DropdownMenu.Root open={open} onOpenChange={setOpen} modal={false}>
        <DropdownMenu.Trigger className="navlink more-btn" data-active={active || undefined} data-active-mid={activeMid || undefined}>
          More<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true"><path d="M6 9l6 6 6-6" /></svg>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content className="menu more-menu" id={id} side={side} align={side === "top" ? "end" : "start"} sideOffset={6} collisionPadding={8}>
          {items.map((n) => (
            <DropdownMenu.Item key={n.href} asChild>
              <Link href={n.href} className={`menu-item${n.mid ? " more-mid" : ""}`} aria-current={isActive(path, n.href) ? "page" : undefined}>{n.label}</Link>
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </div>
  );
}
export const TICKER_RE = /^[A-Za-z0-9][A-Za-z0-9.-]{0,9}$/;

/** The in-app page the user came from (path + query), for "Back" links. */
const PreviousPage = createContext<string | null>(null);
export const usePreviousPage = () => useContext(PreviousPage);

const SECTIONS: [string, string][] = [
  ["/today", "Today"], ["/scanner/custom", "Custom scan"], ["/scanner/history", "Scan history"], ["/scanner", "Scanner"],
  ["/watchlists", "Watchlists"], ["/alerts", "Alerts"], ["/track-record", "Track record"], ["/brief", "Market Brief"],
  ["/account", "Account"], ["/journal", "Journal"], ["/day-trader", "Day Trader"], ["/stocks", "Stock Intelligence"], ["/paper", "Paper trading"], ["/admin", "Admin"],
];

/** "Back to Watchlists" for /watchlists?id=3; null when there is no previous page here. */
export function backLabel(prev: string | null): string | null {
  const base = prev?.split("?")[0];
  if (!base) return null;
  if (base === "/stocks/compare") return "Back to comparison";
  if (base.startsWith("/stocks/")) return "Back to the previous stock";
  const hit = SECTIONS.find(([p]) => base === p || base.startsWith(`${p}/`));
  return hit ? `Back to ${hit[1]}` : "Back";
}

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const [trail, setTrail] = useState<{ prev: string | null; cur: string | null }>({ prev: null, cur: null });
  useEffect(() => {
    // Query strings are read here, not with useSearchParams, so the shell needs no Suspense boundary.
    const here = path + (typeof window === "undefined" ? "" : window.location.search);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- records the route change the router just made
    setTrail((t) => (t.cur && t.cur.split("?")[0] === path ? { ...t, cur: here } : { prev: t.cur, cur: here }));
  }, [path]);
  const { me, signOut } = useSession();
  const [menu, setMenu] = useState(false);
  const closeMenu = useCallback(() => setMenu(false), []);
  useCloseOnNavigate(closeMenu, path);


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
          <StockSearch compact />
          <div className="account">
            <DropdownMenu.Root open={menu} onOpenChange={setMenu} modal={false}>
              <DropdownMenu.Trigger className="btn" aria-label={me ? `${me.plan_label} · Account` : undefined}>
                {me ? <>{me.plan_label}<span className="hide-narrow">&nbsp;· Account</span></> : "Account"}
              </DropdownMenu.Trigger>
              <DropdownMenu.Content className="menu" align="end" sideOffset={6} collisionPadding={8}>
                {me && <DropdownMenu.Label className="cap menu-email">{me.email}</DropdownMenu.Label>}
                <DropdownMenu.Item asChild><Link href="/account" className="menu-item">Account &amp; billing</Link></DropdownMenu.Item>
                <DropdownMenu.Item asChild><Link href="/journal" className="menu-item">Journal</Link></DropdownMenu.Item>
                <DropdownMenu.Item asChild><Link href="/account#invite-h" className="menu-item">Invite a friend</Link></DropdownMenu.Item>
                <DropdownMenu.Item asChild><Link href="/paper" className="menu-item">Paper trading</Link></DropdownMenu.Item>
                <DropdownMenu.Item asChild><Link href="/pricing" className="menu-item">Plans and pricing</Link></DropdownMenu.Item>
                {me?.is_admin && <DropdownMenu.Item asChild><Link href="/admin" className="menu-item">Admin</Link></DropdownMenu.Item>}
                <DropdownMenu.Item className="menu-item" onSelect={() => void signOut()}>Sign out</DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          </div>
        </div>
      </header>
      <nav aria-label="Sections" className="tabbar">
        {NAV.filter((n) => n.tab).map((n) => (
          <Link key={n.href} href={n.href} className="tab" aria-current={isActive(path, n.href) ? "page" : undefined}>{n.label}</Link>
        ))}
        <MoreMenu items={NAV.filter((n) => !n.tab).map(({ href, label }) => ({ href, label }))} path={path} id="more-tab" className="tab-more" side="top" />
      </nav>
      <InstallHint />
      <main id="main" className="main"><PreviousPage.Provider value={trail.prev}>{children}</PreviousPage.Provider></main>
      <footer className="foot">
        <p className="cap">HSF Score is an opportunity ranking, not a probability of profit. Educational research only, not financial advice. <Link href="/how-hsf-works">How HSF works</Link></p>
        <p className="cap">Data: scheduled HSF scans · Alpaca. Setup prices come from the latest scan; SPY, QQQ and the price strip are quotes refreshed every few minutes.</p>
      </footer>
    </div>
  );
}
