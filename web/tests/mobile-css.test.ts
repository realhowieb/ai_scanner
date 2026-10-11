import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const css = readFileSync(join(__dirname, "..", "src", "app", "globals.css"), "utf8");

describe("mobile CSS polish", () => {
  it("keeps the phone header to one row, with sections in a bottom tab bar", () => {
    expect(css).toContain("@media (max-width: 760px)");
    expect(css).toContain(".topbar-in { padding: 8px 16px; gap: 10px; flex-wrap: nowrap; }");
    expect(css).toContain(".tabbar { display: flex; position: fixed;");
  });

  it("keeps the desktop header on one row", () => {
    expect(css).toContain(".topbar-in { flex-wrap: nowrap; }");
  });

  it("makes core phone actions and filters comfortable to tap", () => {
    expect(css).toContain(".inline-field { width: 100%; align-items: flex-start; flex-direction: column;");
    expect(css).toContain(".inline-field select { width: 100%; }");
    expect(css).toContain(".priority-steps .btn, .workflow-item .btn, .row-actions .btn-primary { flex: 1 1 100%; }");
    expect(css).toContain(".btn { min-height: 42px; white-space: normal; text-align: center; }");
  });

  it("keeps narrow account and status content from squeezing sideways", () => {
    expect(css).toContain(".kv > div { align-items: flex-start; flex-direction: column;");
    expect(css).toContain(".status-strip { align-items: flex-start; flex-direction: column; }");
    expect(css).toContain(".tiles { grid-template-columns: minmax(0, 1fr); }");
  });

  it("orders Stock Intelligence for phone scanning", () => {
    expect(css).toContain(".stock-split .score-card { order: -1; }");
    expect(css).toContain(".stock-split .price-card { order: 2; }");
    expect(css).toContain(".snapshot-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }");
  });
  it("keeps phone fields from zooming and small links easy to tap", () => {
    expect(css).toContain("input, select, textarea { font-size: 16px; }");
    expect(css).toContain(".link-btn { padding: 8px 4px; margin: -8px -4px; }");
    expect(css).toContain(".search input { background: transparent; border: 0; outline: none; color: var(--text); width: 100%; min-width: 0; align-self: stretch;");
  });

  it("lays out the phone review queue and market tiles compactly", () => {
    expect(css).toContain(".review-queue .row { display: grid; grid-template-columns: minmax(0, 1fr) auto;");
    expect(css).toContain(".tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 16px; }");
    expect(css).toContain(".wl-act { display: inline-flex;");
  });
});
