"use client";

import { useId, useState } from "react";
import { useRouter } from "next/navigation";

import { api, unwrap } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { readCachedMe } from "@/session/meCache";

const SYMBOL = /^[A-Z][A-Z0-9.\-]{0,9}$/;

export function StockSearch({ compact = false }: { compact?: boolean }) {
  const router = useRouter();
  const owner = () => readCachedMe()?.email ?? "";
  const id = useId();
  const [q, setQ] = useState("");
  const [focused, setFocused] = useState(false);
  const [bad, setBad] = useState(false);
  const [recent, setRecent] = useState<string[]>([]);
  const [index, setIndex] = useState(-1);
  const text = q.trim();
  // Debounce at the request boundary, with AbortSignal to drop superseded queries.
  const suggestions = useApi(focused && text ? `symbols:${text}` : null, async (signal) => {
    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(resolve, 250);
      signal.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("Aborted", "AbortError")); }, { once: true });
    });
    return unwrap(api.GET("/v1/symbols/search", { params: { query: { q: text } }, signal }));
  });
  const options = text ? suggestions.data?.results ?? [] : recent.map((ticker) => ({ ticker, name: "Recent search" }));
  const open = (ticker: string) => {
    const next = [ticker, ...recent.filter((t) => t !== ticker)].slice(0, 6);
    try { sessionStorage.setItem(`hsf-search:${owner()}`, JSON.stringify(next)); } catch { /* storage optional */ }
    setRecent(next); setQ(""); setFocused(false); setBad(false); setIndex(-1);
    router.push(`/stocks/${encodeURIComponent(ticker)}`);
  };
  return <form className={`stock-search ${compact ? "search" : "stock-inline-search"}`} role="search" onSubmit={(e) => {
    e.preventDefault();
    if (index >= 0 && options[index]) return open(options[index].ticker);
    const exact = options.find((o) => o.ticker === text.toUpperCase());
    if (exact) return open(exact.ticker);
    if (options.length === 1 && options[0]) return open(options[0].ticker);
    const companyMatch = options.some((o) => o.name.toLowerCase().includes(text.toLowerCase()) && !o.ticker.startsWith(text.toUpperCase()));
    if (!companyMatch && SYMBOL.test(text.toUpperCase())) return open(text.toUpperCase());
    setBad(true); setFocused(true);
  }}>
    <label className="sr-only" htmlFor={id}>Search ticker or company</label>
    <span className={compact ? "search-input" : "field grow"}><input id={id} type="search" placeholder="Ticker or company" maxLength={80} value={q} autoComplete="off"
      role="combobox" aria-expanded={focused && options.length > 0} aria-controls={`${id}-options`}
      aria-autocomplete="list" aria-activedescendant={index >= 0 && options[index] ? `${id}-${index}` : undefined}
      aria-invalid={bad || undefined} aria-describedby={bad ? `${id}-error` : undefined}
      onFocus={() => {
        setFocused(true);
        try {
          const saved: unknown = JSON.parse(sessionStorage.getItem(`hsf-search:${owner()}`) || "[]");
          setRecent(Array.isArray(saved) ? saved.filter((t): t is string => typeof t === "string" && SYMBOL.test(t)).slice(0, 6) : []);
        } catch { setRecent([]); }
      }}
      onBlur={(e) => { if (!e.currentTarget.form?.contains(e.relatedTarget as Node)) setFocused(false); }}
      onChange={(e) => { setQ(e.target.value); setFocused(true); setBad(false); setIndex(-1); }}
      onKeyDown={(e) => {
        if (e.key === "Escape") { setFocused(false); setIndex(-1); }
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          e.preventDefault(); setFocused(true);
          setIndex((i) => !options.length ? -1 : i < 0 ? e.key === "ArrowDown" ? 0 : options.length - 1 : (i + (e.key === "ArrowDown" ? 1 : -1) + options.length) % options.length);
        }
      }} /></span>
    {!compact && <button className="btn btn-primary" type="submit">Open</button>}
    {focused && options.length > 0 && <ul id={`${id}-options`} className="stock-suggestions" role="listbox">
      {options.map((o, i) => <li key={o.ticker} id={`${id}-${i}`} role="option" aria-selected={i === index}>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => open(o.ticker)}><strong>{o.ticker}</strong> <span>{o.name}</span></button>
      </li>)}
    </ul>}
    {bad && <span className="form-error" id={`${id}-error`} role="alert">Choose a suggested company or enter a ticker like AAPL.</span>}
    {focused && text && (suggestions.error || suggestions.data?.available === false) && <span className="cap">Company suggestions are unavailable. You can still open a ticker.</span>}
  </form>;
}
