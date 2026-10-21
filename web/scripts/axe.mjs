// Runs axe-core (automated accessibility rules, WCAG 2.1 A + AA) in a Playwright page.
// Returns the violations, each with its impact and up to three offending elements.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const AXE_SOURCE = readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");
export const BLOCKING = new Set(["critical", "serious"]);

export async function runAxe(page) {
  if (!(await page.evaluate(() => Boolean(window.axe)))) await page.addScriptTag({ content: AXE_SOURCE });
  return page.evaluate(async () => {
    const res = await window.axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] }, resultTypes: ["violations"] });
    return res.violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, count: v.nodes.length, targets: v.nodes.slice(0, 3).map((n) => n.target.join(" ")) }));
  });
}

/** One line per rule, worst first. */
export function describe(violations) {
  const order = ["critical", "serious", "moderate", "minor"];
  return [...violations].sort((a, b) => order.indexOf(a.impact) - order.indexOf(b.impact))
    .map((v) => `${v.impact} ${v.id} ×${v.count}: ${v.help} [${v.targets.join(" | ")}]`);
}
