import { STALE_BUILD_SCRIPT } from "@/lib/staleBuild";

// Runs the inline script against a fake window so its listeners can be fired by hand.
function boot(stored: string | null = null) {
  const listeners: Record<string, ((e: unknown) => void)[]> = {};
  const store = new Map<string, string>(stored ? [["hsf-stale-reload", stored]] : []);
  const reload = vi.fn();
  const links: { href: string; sheet: unknown }[] = [];
  const win = {
    addEventListener: (t: string, fn: (e: unknown) => void) => (listeners[t] ||= []).push(fn),
  };
  const sessionStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => store.set(k, v) };
  const document = { querySelectorAll: () => links };
  new Function("window", "location", "sessionStorage", "document", STALE_BUILD_SCRIPT)(win, { reload }, sessionStorage, document);
  const fire = (t: string, e: unknown) => listeners[t]?.forEach((fn) => fn(e));
  return { fire, reload, links, store };
}

describe("stale build reload", () => {
  it("reloads when a build stylesheet fails to load", () => {
    const { fire, reload } = boot();
    fire("error", { target: { tagName: "LINK", href: "https://x/_next/static/chunks/old.css" } });
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("reloads when a build script or lazy chunk fails", () => {
    const a = boot();
    a.fire("error", { target: { tagName: "SCRIPT", src: "https://x/_next/static/chunks/old.js" } });
    expect(a.reload).toHaveBeenCalledTimes(1);
    const b = boot();
    b.fire("unhandledrejection", { reason: { name: "ChunkLoadError", message: "Loading chunk 12 failed." } });
    expect(b.reload).toHaveBeenCalledTimes(1);
  });

  it("reloads when a build stylesheet never applied", () => {
    const { fire, reload, links } = boot();
    links.push({ href: "https://x/_next/static/chunks/old.css", sheet: null });
    fire("load", {});
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("ignores other failures", () => {
    const { fire, reload, links } = boot();
    fire("error", { target: { tagName: "IMG", src: "https://x/_next/static/media/a.png" } });
    fire("error", { target: { tagName: "SCRIPT", src: "https://cdn.example/x.js" } });
    fire("unhandledrejection", { reason: new Error("network down") });
    links.push({ href: "https://x/_next/static/chunks/ok.css", sheet: {} });
    fire("load", {});
    expect(reload).not.toHaveBeenCalled();
  });

  it("reloads at most once a minute so a missing file cannot loop", () => {
    const recent = boot(String(Date.now() - 5_000));
    recent.fire("error", { target: { tagName: "LINK", href: "/_next/static/chunks/old.css" } });
    expect(recent.reload).not.toHaveBeenCalled();
    const fresh = boot();
    fresh.fire("error", { target: { tagName: "LINK", href: "/_next/static/chunks/old.css" } });
    fresh.fire("error", { target: { tagName: "LINK", href: "/_next/static/chunks/old2.css" } });
    expect(fresh.reload).toHaveBeenCalledTimes(1);
  });
});
