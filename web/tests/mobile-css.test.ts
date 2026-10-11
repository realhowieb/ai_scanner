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
});
