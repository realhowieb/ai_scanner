import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { useApi } from "@/hooks/useApi";
import { writeCachedMe } from "@/session/meCache";
import type { Schemas } from "@/api/client";

function Probe({ load }: { load: () => Promise<string> }) {
  const s = useApi("probe", () => load());
  return <p>{s.loading ? "loading" : `${s.data}${s.refreshing ? " (refreshing)" : ""}`}</p>;
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
