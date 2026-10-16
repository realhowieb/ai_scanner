import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { clearApiCache, useApi } from "@/hooks/useApi";
import { writeCachedMe } from "@/session/meCache";
import type { Schemas } from "@/api/client";

function Probe({ load, cacheKey = "probe" }: { load: () => Promise<string>; cacheKey?: string | null }) {
  const s = useApi(cacheKey, () => load());
  return <><p>{s.loading ? "loading" : `${s.data}${s.refreshing ? " (refreshing)" : ""}`}</p><button onClick={s.reload}>Reload</button></>;
}

function deferred() {
  let resolve: (v: string) => void = () => {};
  const promise = new Promise<string>((r) => (resolve = r));
  return { promise, resolve };
}

describe("useApi page cache", () => {
  it("shows the remembered answer at once when a page comes back, then the fresh one", async () => {
    const first = render(<Probe load={() => Promise.resolve("v1")} />);
    expect(screen.getByText("loading")).toBeInTheDocument();
    await screen.findByText("v1");
    first.unmount();

    const next = deferred();
    render(<Probe load={() => next.promise} />);
    expect(screen.getByText("v1 (refreshing)")).toBeInTheDocument();
    next.resolve("v2");
    await screen.findByText("v2");
  });

  it("forgets everything when the signed-in account changes", async () => {
    writeCachedMe({ email: "a@example.com" } as Schemas["Me"]);
    const first = render(<Probe load={() => Promise.resolve("a's data")} />);
    await screen.findByText("a's data");
    first.unmount();

    writeCachedMe({ email: "b@example.com" } as Schemas["Me"]);
    const next = deferred();
    render(<Probe load={() => next.promise} />);
    expect(screen.getByText("loading")).toBeInTheDocument();
    next.resolve("b's data");
    await waitFor(() => expect(screen.getByText("b's data")).toBeInTheDocument());
  });
});

describe("useApi request isolation", () => {
  it("ignores a cancelled response even when the loader ignores abort", async () => {
    const old = deferred();
    const first = render(<Probe load={() => old.promise} />);
    first.unmount();
    const fresh = render(<Probe load={() => Promise.resolve("fresh")} />);
    await screen.findByText("fresh");
    await act(async () => { old.resolve("stale"); });
    fresh.unmount();
    render(<Probe load={() => deferred().promise} />);
    expect(screen.getByText("fresh (refreshing)")).toBeInTheDocument();
  });

  it("discards old responses and visible data across account changes", async () => {
    writeCachedMe({ email: "a@example.com" } as Schemas["Me"]);
    const old = deferred();
    const view = render(<Probe load={() => old.promise} />);
    const next = deferred();
    view.rerender(<Probe load={() => next.promise} />);
    act(() => writeCachedMe({ email: "b@example.com" } as Schemas["Me"]));
    await act(async () => { old.resolve("private a data"); });
    expect(screen.getByText("loading")).toBeInTheDocument();
    next.resolve("private b data");
    await screen.findByText("private b data");
    act(() => writeCachedMe({ email: "c@example.com" } as Schemas["Me"]));
    expect(screen.queryByText("private b data")).not.toBeInTheDocument();
  });

  it("does not show another key's data when the new request fails", async () => {
    const view = render(<Probe cacheKey="first" load={() => Promise.resolve("first data")} />);
    await screen.findByText("first data");
    view.rerender(<Probe cacheKey="second" load={() => Promise.reject(new Error("offline"))} />);
    expect(screen.queryByText("first data")).not.toBeInTheDocument();
    await screen.findByText("null");
    view.rerender(<Probe cacheKey={null} load={() => Promise.resolve("unused")} />);
    expect(screen.getByText("null")).toBeInTheDocument();
  });

  it("invalidates in-flight answers when the cache is explicitly cleared", async () => {
    const old = deferred();
    const view = render(<Probe load={() => old.promise} />);
    const next = deferred();
    view.rerender(<Probe load={() => next.promise} />);
    act(() => clearApiCache());
    await act(async () => { old.resolve("invalidated"); });
    expect(screen.getByText("loading")).toBeInTheDocument();
    next.resolve("updated");
    await screen.findByText("updated");
  });
});


describe("useApi refresh recovery", () => {
  it("keeps remembered data when background revalidation fails", async () => {
    const first = render(<Probe load={() => Promise.resolve("remembered")} />);
    await screen.findByText("remembered");
    first.rerender(<Probe load={() => Promise.reject(new Error("offline"))} />);
    fireEvent.click(screen.getByText("Reload"));
    await waitFor(() => expect(screen.getByText("remembered")).toBeInTheDocument());
  });

  it("ignores a superseded reload response", async () => {
    const old = deferred();
    const view = render(<Probe load={() => old.promise} />);
    const next = deferred();
    view.rerender(<Probe load={() => next.promise} />);
    fireEvent.click(screen.getByText("Reload"));
    next.resolve("latest");
    await screen.findByText("latest");
    await act(async () => { old.resolve("superseded"); });
    expect(screen.getByText("latest")).toBeInTheDocument();
  });
});
