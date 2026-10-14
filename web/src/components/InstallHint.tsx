"use client";

import { useState, useSyncExternalStore } from "react";

import { dismissHint, hintDismissed, isIPhone, isStandalone } from "@/lib/homeScreen";

const noSubscribe = () => () => undefined;

/** iPhone Safari only: how to add HSF to the Home Screen, which is where iPhones allow alerts. */
export function InstallHint() {
  // Read on the client only: the server can't know the device, so it renders nothing.
  const eligible = useSyncExternalStore(noSubscribe, () => isIPhone() && !isStandalone() && !hintDismissed(), () => false);
  const [closed, setClosed] = useState(false);
  if (!eligible || closed) return null;
  return (
    <aside className="install-hint" aria-label="Add HSF to your Home Screen">
      <p>
        <b>Get HSF as an app.</b> Tap <span className="share-glyph" aria-label="Share">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 3v12M7 8l5-5 5 5" /><path d="M5 12v8h14v-8" /></svg>
        </span> Share, then <b>Add to Home Screen</b>. Alerts on iPhone only work from there.
      </p>
      <button type="button" className="btn" onClick={() => { dismissHint(); setClosed(true); }}>Not now</button>
    </aside>
  );
}
