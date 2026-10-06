"use client";

// "Save to watchlist" for one ticker: pick a list (or make one), then the server says
// whether it was added or already there. Nothing is shown as saved until it answers.
import { useState } from "react";
import type { FormEvent } from "react";

import { watchlists } from "@/api/userData";
import { Dialog } from "@/components/Dialog";
import { ErrorLine, Pill } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";

export function SaveToWatchlistDialog({ ticker, open, onClose, onSaved, inLists = [] }: {
  ticker: string; open: boolean; onClose: () => void; onSaved?: () => void; inLists?: number[];
}) {
  const lists = useApi(open ? `wl-picker:${ticker}` : null, (signal) => watchlists.list(signal));
  const act = useAction();
  const [done, setDone] = useState<Record<number, "added" | "already">>({});
  const [name, setName] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const add = async (id: number, listName: string) => {
    const r = await act.run(() => watchlists.addTickers(id, [ticker]));
    if (!r) return;
    if (r.invalid.length) {
      setMessage(`${ticker} isn't a valid ticker symbol.`);
      return;
    }
    const added = r.added.includes(ticker);
    setDone((d) => ({ ...d, [id]: added ? "added" : "already" }));
    setMessage(added ? `Saved ${ticker} to ${listName}.` : `${ticker} is already in ${listName}.`);
    if (added) onSaved?.();
  };

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const n = name.trim();
    if (!n) return;
    const wl = await act.run(() => watchlists.create(n));
    if (!wl) return;
    setName("");
    lists.reload();
    await add(wl.id, wl.name);
  };

  const close = () => {
    setDone({});
    setMessage(null);
    act.clear();
    onClose();
  };

  const items = lists.data ?? [];
  return (
    <Dialog open={open} title={`Save ${ticker} to a watchlist`} onClose={close}
      footer={<button type="button" className="btn" onClick={close}>Done</button>}>
      {lists.loading && !lists.data ? <p className="cap">Loading your watchlists…</p> : lists.error ? <ErrorLine error={lists.error} /> : (
        <div className="stack-sm">
          {items.length === 0 && <p className="cap">You don&apos;t have a watchlist yet. Name one below and {ticker} goes straight in.</p>}
          <ul className="pick-list">
            {items.map((w) => {
              const state = done[w.id] ?? (inLists.includes(w.id) ? "already" : undefined);
              return (
                <li key={w.id}>
                  <button type="button" className="pick" onClick={() => void add(w.id, w.name)} disabled={act.busy || !!state}>
                    <span className="strong">{w.name}</span>
                    <span className="cap">{w.symbol_count} ticker{w.symbol_count === 1 ? "" : "s"}</span>
                    {w.is_default && <Pill>Default</Pill>}
                    <span className="grow" />
                    <span className="cap">{state === "added" ? "Saved" : state === "already" ? "Already in it" : "Add"}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          <form className="inline-form" onSubmit={create}>
            <label className="field grow"><span>New watchlist</span>
              <input value={name} maxLength={80} onChange={(e) => setName(e.target.value)} placeholder="e.g. Breakouts" />
            </label>
            <button type="submit" className="btn" disabled={act.busy || !name.trim()}>Create and save</button>
          </form>
          {message && <p className="notice" role="status">{message}</p>}
          <ErrorLine error={act.error} />
        </div>
      )}
    </Dialog>
  );
}

/** A button that opens the dialog. */
export function SaveToWatchlistButton({ ticker, compact = false, inLists, onSaved }: {
  ticker: string; compact?: boolean; inLists?: number[]; onSaved?: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      {compact ? (
        <button type="button" className="icon-btn" aria-label={`Save ${ticker} to a watchlist`} onClick={() => setOpen(true)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>
        </button>
      ) : (
        <button type="button" className="btn" onClick={() => setOpen(true)}>Save to watchlist</button>
      )}
      <SaveToWatchlistDialog ticker={ticker} open={open} onClose={() => setOpen(false)} inLists={inLists} onSaved={onSaved} />
    </>
  );
}
