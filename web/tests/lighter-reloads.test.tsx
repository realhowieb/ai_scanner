import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useApi } from "@/hooks/useApi";

vi.mock("next/link", () => ({
  default: ({ prefetch, children, ...rest }: { prefetch?: boolean | null; children: React.ReactNode; href: string }) => (
    <a {...rest} data-prefetch={String(prefetch)}>{children}</a>
  ),
}));

import Link from "@/components/Link";

function Probe({ load, name }: { load: (signal: AbortSignal) => Promise<string>; name: string }) {
  const s = useApi("shared-key", load);
  return <><p>{name}: {s.loading ? "loading" : s.data}</p><button onClick={s.reload}>Reload {name}</button></>;
}

function deferred() {
  let resolve: (v: string) => void = () => {};
  const promise = new Promise<string>((r) => (resolve = r));
  return { promise, resolve };
}

describe("one request per key at a time", () => {
  it("components asking for the same key together share one request", async () => {
    const d = deferred();
    const load = vi.fn(() => d.promise);
    render(<><Probe name="a" load={load} /><Probe name="b" load={load} /><Probe name="c" load={load} /></>);
    expect(load).toHaveBeenCalledTimes(1);
    d.resolve("rows");
    expect(await screen.findByText("a: rows")).toBeInTheDocument();
    expect(screen.getByText("c: rows")).toBeInTheDocument();
  });

  it("keeps the shared request alive while anyone still waits, and a reload asks again", async () => {
    const d = deferred();
    let signal: AbortSignal | null = null;
    const load = vi.fn((s: AbortSignal) => { signal = s; return d.promise; });
    const one = render(<Probe name="a" load={load} />);
    render(<Probe name="b" load={load} />);
    one.unmount();
    expect(signal!.aborted).toBe(false);
    d.resolve("kept");
    await screen.findByText("b: kept");

    const again = vi.fn(() => Promise.resolve("fresh"));
    render(<Probe name="c" load={again} />);
    fireEvent.click(screen.getByText("Reload c"));
    await screen.findByText("c: fresh");
    expect(again).toHaveBeenCalledTimes(2);
  });

  it("cancels the request once nobody waits for it", () => {
    let signal: AbortSignal | null = null;
    const one = render(<Probe name="a" load={(s) => { signal = s; return deferred().promise; }} />);
    one.unmount();
    expect(signal!.aborted).toBe(true);
  });
});

describe("links prefetch on intent", () => {
  it("does not prefetch until hovered, touched or focused", () => {
    render(<><Link href="/scanner">Scanner</Link><Link href="/today" prefetch>Today</Link></>);
    const scanner = screen.getByText("Scanner");
    expect(scanner.dataset.prefetch).toBe("false");
    fireEvent.mouseEnter(scanner);
    expect(scanner.dataset.prefetch).toBe("null");
    expect(screen.getByText("Today").dataset.prefetch).toBe("true");
  });
});
