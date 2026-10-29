"use client";

import Link from "@/components/Link";
import { usePathname } from "next/navigation";
import { DropdownMenu } from "radix-ui";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

import { AlertsBadge, useUnreadAlerts } from "@/components/AlertsBadge";
import { FeedbackDialog } from "@/components/FeedbackDialog";
import { StockSearch } from "@/components/StockSearch";
import { BillingBanner } from "@/components/BillingBanner";
import { InstallHint } from "@/components/InstallHint";
import { MarketStatus } from "@/components/MarketStatus";
import { NavIcon } from "@/components/NavIcon";
import { WhatsNew } from "@/components/WhatsNew";
import { FOUNDER_HREF, usePublicFounding } from "@/features/FoundingBanner";
import { useSession } from "@/session/SessionProvider";

/** Desktop sidebar groups (every section, always visible). `tab`: in the phone tab bar
 * (the rest sit under the tab bar's More). */
const NAV = [
  { href: "/today", label: "Today", icon: "today", group: "Markets", tab: true },
  { href: "/brief", label: "Brief", icon: "brief", group: "Markets", tab: false },
  { href: "/scanner", label: "Scanner", icon: "scanner", group: "Markets", tab: true },
  { href: "/day-trader", label: "Day Trader", icon: "daytrader", group: "Markets", tab: false },
  { href: "/stocks", label: "Stocks", icon: "stocks", group: "Research", tab: false },
  { href: "/watchlists", label: "Watchlists", icon: "watchlists", group: "Research", tab: true },
  { href: "/alerts", label: "Alerts", icon: "alerts", group: "Research", tab: true },
  { href: "/track-record", label: "Track record", icon: "track", group: "Research", tab: false },
  { href: "/ask", label: "Ask HSF", icon: "ask", group: "Tools", tab: false },
  { href: "/journal", label: "Journal", icon: "journal", group: "Tools", tab: false },
  { href: "/paper", label: "Paper trading", icon: "paper", group: "Tools", tab: false },
];
const GROUPS = ["Markets", "Research", "Tools"];

const isActive = (path: string, href: string) => path === href || path.startsWith(`${href}/`);

/** Keeps a controlled menu closed after the route changes. */
function useCloseOnNavigate(close: () => void, path: string) {
  // eslint-disable-next-line react-hooks/exhaustive-deps -- close on navigation only
  useEffect(() => close(), [path]);
}

type NavItem = { href: string; label: string; icon?: string };

// Menus use Radix DropdownMenu: arrow keys, Home/End and type-ahead move between items,
// Escape closes and returns focus to the button. Rendered in place (no portal) so the
// menu stays inside its nav landmark; modal={false} keeps the page usable while it is open.
function MoreMenu({ items, path, id, className, side = "bottom" }: { items: NavItem[]; path: string; id: string; className: string; side?: "top" | "bottom" }) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  useCloseOnNavigate(close, path);
  const active = items.some((n) => isActive(path, n.href));
  return (
    <div className={`more ${className}`}>
      <DropdownMenu.Root open={open} onOpenChange={setOpen} modal={false}>
        <DropdownMenu.Trigger className="navlink more-btn" data-active={active || undefined}>
          <NavIcon name="more" /><span className="tab-label">More</span>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content className="menu more-menu" id={id} side={side} align={side === "top" ? "end" : "start"} sideOffset={6} collisionPadding={8}>
          {items.map((n) => (
            <DropdownMenu.Item key={n.href} asChild>
              <Link href={n.href} className="menu-item" aria-current={isActive(path, n.href) ? "page" : undefined}>{n.label}</Link>
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
  ["/account", "Account"], ["/ask", "Ask HSF"], ["/journal", "Journal"], ["/day-trader", "Day Trader"], ["/stocks", "Stock Intelligence"], ["/paper", "Paper trading"], ["/admin", "Admin"],
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
  const founding = usePublicFounding();
  const closeMenu = useCallback(() => setMenu(false), []);
  useCloseOnNavigate(closeMenu, path);
  const unread = useUnreadAlerts(path, !!me);
  const [feedback, setFeedback] = useState(false);

  const plan = me?.plan;
  return (
    <div className="shell">
      <a href="#main" className="skip">Skip to content</a>
      <aside className="sidebar">
        <Link href="/today" className="brand side-brand">HSFinest<span>.AI</span></Link>
        <nav aria-label="Main" className="side-nav">
          {GROUPS.map((g) => (
            <div key={g} className="side-group">
              <p className="side-group-label" aria-hidden="true">{g}</p>
              <ul>
                {NAV.filter((n) => n.group === g).map((n) => (
                  <li key={n.href}>
                    <Link href={n.href} className="side-link" aria-current={isActive(path, n.href) ? "page" : undefined} title={n.label}>
                      <NavIcon name={n.icon} /><span className="side-label">{n.label}</span>{n.href === "/alerts" && <AlertsBadge count={unread} />}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
        <div className="side-foot">
          {plan && plan !== "premium" && plan !== "admin" && (
            <Link href="/pricing" className="side-upgrade">
              <span className="strong">{plan === "pro" ? "Go Premium" : "Upgrade"}</span>
              <span className="cap">{plan === "pro" ? "AI research tools, PreBreakout" : "Pro from $25/mo"}</span>
            </Link>
          )}
          <Link href="/how-hsf-works" className="side-help cap">How HSF works</Link>
        </div>
      </aside>
      <div className="frame">
      <header className="topbar">
        <div className="topbar-in">
          <Link href="/today" className="brand top-brand">HSFinest<span>.AI</span></Link>
          <StockSearch compact />
          <MarketStatus />
          {me && <span className={`plan-badge plan-${me.plan}`} aria-hidden="true">{me.plan_label}</span>}
          <div className="account">
            <DropdownMenu.Root open={menu} onOpenChange={setMenu} modal={false}>
              <DropdownMenu.Trigger className="btn account-btn" aria-label={me ? `${me.plan_label} · Account` : undefined}>
                <span className="avatar" aria-hidden="true">{(me?.name || me?.email || "?").trim().charAt(0).toUpperCase()}</span>
                <span className="account-word">Account</span>
              </DropdownMenu.Trigger>
              <DropdownMenu.Content className="menu" align="end" sideOffset={6} collisionPadding={8}>
                {me && <DropdownMenu.Label className="cap menu-email">{me.email}<br /><span className="menu-plan">{me.plan_label} plan</span></DropdownMenu.Label>}
                <DropdownMenu.Item asChild><Link href="/account" className="menu-item">Account &amp; billing</Link></DropdownMenu.Item>
                <DropdownMenu.Item asChild><Link href="/account#invite-h" className="menu-item">Invite a friend</Link></DropdownMenu.Item>
                <DropdownMenu.Item asChild><Link href="/pricing" className="menu-item">Plans and pricing</Link></DropdownMenu.Item>
                {founding && <DropdownMenu.Item asChild><Link href={FOUNDER_HREF} className="menu-item">Founding 100</Link></DropdownMenu.Item>}
                <DropdownMenu.Item className="menu-item" onSelect={() => setFeedback(true)}>Send feedback</DropdownMenu.Item>
                {me?.is_admin && <DropdownMenu.Item asChild><Link href="/admin" className="menu-item">Admin</Link></DropdownMenu.Item>}
                <DropdownMenu.Item className="menu-item" onSelect={() => void signOut()}>Sign out</DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          </div>
        </div>
      </header>
      <nav aria-label="Sections" className="tabbar">
        {NAV.filter((n) => n.tab).map((n) => (
          <Link key={n.href} href={n.href} className="tab" aria-current={isActive(path, n.href) ? "page" : undefined}><NavIcon name={n.icon} /><span className="tab-label">{n.label}</span>{n.href === "/alerts" && <AlertsBadge count={unread} />}</Link>
        ))}
        <MoreMenu items={NAV.filter((n) => !n.tab)} path={path} id="more-tab" className="tab-more" side="top" />
      </nav>
      <BillingBanner />
      <InstallHint />
      <WhatsNew />
      <FeedbackDialog open={feedback} onClose={() => setFeedback(false)} />
      <main id="main" className="main"><PreviousPage.Provider value={trail.prev}>{children}</PreviousPage.Provider></main>
      <footer className="foot">
        <p className="cap">HSF Score is an opportunity ranking, not a probability of profit. Educational research only, not financial advice. <Link href="/how-hsf-works">How HSF works</Link> · <button type="button" className="link-btn cap" onClick={() => setFeedback(true)}>Send feedback</button></p>
        <p className="cap">Data: scheduled HSF scans · Alpaca. Setup prices come from the latest scan; SPY, QQQ and the price strip are quotes refreshed every few minutes.</p>
      </footer>
      </div>
    </div>
  );
}
