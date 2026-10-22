// After `next build`: upload the browser source maps to Sentry, then delete them so
// they're never served. Runs only when SENTRY_AUTH_TOKEN is set on the web service
// (next.config.ts makes the maps only then). Never fails the build.
//
//   SENTRY_AUTH_TOKEN   an organization token with project:releases (Sentry → Settings → Auth Tokens)
//   SENTRY_ORG          default hs-finest-ai
//   SENTRY_PROJECT      default python (the project the API's SENTRY_DSN reports to)
//
// Crash reports from the browser reach Sentry through the API (POST /v1/client-errors),
// tagged release web@<commit>; the maps are uploaded under that release, so Sentry shows
// the original file, line and function instead of minified chunks.
import { execFileSync } from "node:child_process";
import { readdirSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";

const STATIC = ".next/static";

function maps(dir) {
  let out = [];
  let entries = [];
  try { entries = readdirSync(dir); } catch { return out; }
  for (const name of entries) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out = out.concat(maps(p));
    else if (name.endsWith(".js.map") || name.endsWith(".css.map")) out.push(p);
  }
  return out;
}

const token = process.env.SENTRY_AUTH_TOKEN?.trim();
const release = process.env.RENDER_GIT_COMMIT?.trim() || process.env.HSF_RELEASE?.trim();
const found = maps(STATIC);

if (token && release && found.length) {
  const org = process.env.SENTRY_ORG?.trim() || "hs-finest-ai";
  const project = process.env.SENTRY_PROJECT?.trim() || "python";
  try {
    execFileSync("npx", ["--no-install", "sentry-cli", "sourcemaps", "upload",
      "--org", org, "--project", project, "--release", `web@${release}`,
      "--url-prefix", "~/_next/static", "--validate", STATIC], { stdio: "inherit", env: process.env });
    console.log(`Sentry: uploaded ${found.length} source maps for web@${release.slice(0, 12)}.`);
  } catch {
    console.warn("Sentry: source map upload failed; the build continues without symbolicated web errors.");
  }
} else if (token && !release) {
  console.warn("Sentry: no RENDER_GIT_COMMIT or HSF_RELEASE, so source maps can't be matched to a release; skipped.");
}

for (const f of found) rmSync(f, { force: true });
if (found.length) console.log(`Removed ${found.length} source maps from the public build.`);
