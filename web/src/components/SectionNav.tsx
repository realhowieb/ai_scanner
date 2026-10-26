"use client";

import { useEffect, useState } from "react";

/** In-page jump links for long research pages (the stock page). A link shows only once its
 * section is on the page, since some sections load later or don't apply to every ticker. */
export function SectionNav({ items, label = "On this page" }: { items: [id: string, label: string][]; label?: string }) {
  const [present, setPresent] = useState<string[]>([]);
  const key = items.map(([id]) => id).join(",");
  useEffect(() => {
    const ids = key.split(",");
    const check = () => {
      const next = ids.filter((id) => document.getElementById(id));
      setPresent((prev) => (prev.join(",") === next.join(",") ? prev : next));
    };
    check();
    const root = document.getElementById("main") ?? document.body;
    const obs = new MutationObserver(check);
    obs.observe(root, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, [key]);
  const shown = items.filter(([id]) => present.includes(id));
  if (shown.length < 2) return null;
  return (
    <nav className="section-nav" aria-label={label}>
      {shown.map(([id, text]) => <a key={id} href={`#${id}`} className="section-link">{text}</a>)}
    </nav>
  );
}
