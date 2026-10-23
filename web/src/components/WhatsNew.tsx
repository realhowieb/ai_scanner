"use client";

import { useState, useSyncExternalStore } from "react";

import { WHATS_NEW } from "@/content/whatsNew";

const KEY = "hsf-whats-new-seen";
const noSubscribe = () => () => undefined;

function seen(): string | null {
  try { return localStorage.getItem(KEY); } catch { return "unavailable"; }
}

/** The newest "What's new" entry, once per browser until dismissed. A first visit only
 * records the entry, so new accounts aren't greeted with old news. */
export function WhatsNew() {
  const latest = WHATS_NEW[0];
  const last = useSyncExternalStore(noSubscribe, seen, () => "server");
  const [closed, setClosed] = useState(false);
  if (!latest || closed || last === "server" || last === "unavailable") return null;
  if (last === null) {
    try { localStorage.setItem(KEY, latest.id); } catch { /* private mode */ }
    return null;
  }
  if (last >= latest.id) return null;
  const close = () => {
    try { localStorage.setItem(KEY, latest.id); } catch { /* private mode */ }
    setClosed(true);
  };
  return (
    <aside className="install-hint whats-new" aria-labelledby="whats-new-h">
      <div>
        <p id="whats-new-h"><b>{latest.title}</b></p>
        <ul>{latest.items.map((t) => <li key={t}>{t}</li>)}</ul>
      </div>
      <button type="button" className="btn" onClick={close}>Got it</button>
    </aside>
  );
}
