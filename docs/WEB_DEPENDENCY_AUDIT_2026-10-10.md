# Web Dependency Audit — 2026-10-10

## Summary

`npm audit --omit=dev` is clean: no production dependency vulnerabilities.

The full `npm audit` reports five high-severity findings, all from one dev-only
lint dependency chain:

`eslint-config-next -> @next/eslint-plugin-next -> fast-glob -> micromatch -> braces`

The underlying advisory is `braces` stack-exhaustion denial of service through
deeply nested patterns (`GHSA-vfj7-8cjw-p6xm`). The installed version is
`braces@3.0.3`, which is also the latest version currently published to npm.

## Decision

Do not run `npm audit fix --force` for this finding right now. npm suggests
changing `eslint-config-next` to `14.2.35`, which would be a major downgrade from
the Next 16 web app tooling and is higher risk than the dev-only advisory.

## Follow-up

Recheck after either:

- `braces` publishes a fixed version, or
- `eslint-config-next` / `@next/eslint-plugin-next` moves to a dependency chain
  that no longer includes the vulnerable `braces` range.

Verification on this pass:

- `npm audit --omit=dev` passed with 0 vulnerabilities
- `npm test` passed: 181 tests
- `npm run build` passed
- `npm run api:check` passed
