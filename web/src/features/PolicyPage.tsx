// One Founding 100 policy page (Terms, Refund Policy or Privacy Policy), rendered on the
// server from the API's text. A proposed text says so at the top: it is not yet reviewed.
import Link from "@/components/Link";
import { PublicFooter, PublicHeader } from "@/features/Landing";
import type { LoadedPolicy } from "@/server/policy";

export const POLICY_LINKS = [
  { type: "founding_terms", href: "/founding-100/terms", label: "Founding Membership Terms" },
  { type: "founding_refunds", href: "/founding-100/refunds", label: "Refund Policy" },
  { type: "privacy", href: "/privacy", label: "Privacy Policy" },
] as const;

function slug(heading: string, i: number): string {
  return `s${i}-${heading.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40)}`;
}

export function PolicyPage({ loaded }: { loaded: LoadedPolicy | null }) {
  if (!loaded) {
    return (
      <div className="pub"><PublicHeader /><main id="main" className="doc">
        <article className="card stack-sm">
          <h1 className="h1">Not published yet</h1>
          <p className="body-sm">This page isn&apos;t available yet. Please check back soon.</p>
          <p className="cap"><Link href="/">← Back to HSF</Link></p>
        </article>
      </main><PublicFooter /></div>
    );
  }
  const { policy, preview } = loaded;
  const headed = policy.sections.map((s, i) => ({ ...s, id: s.heading ? slug(s.heading, i) : undefined }));
  return (
    <div className="pub"><PublicHeader /><main id="main" className="doc">
      <article className="card stack-sm policy">
        {preview && <p className="banner" role="note">Admin preview: this page is not public yet (FOUNDING_LEGAL_PAGES_ENABLED is off).</p>}
        {policy.status === "proposed" && (
          <p className="banner" role="note">Proposed terms, pending legal review. Not yet in effect.</p>
        )}
        <h1 className="h1">{policy.title}</h1>
        <p className="cap">Version {policy.version} · {policy.effective_date}</p>
        {headed.some((s) => s.id) && (
          <nav aria-label="On this page" className="chips">
            {headed.filter((s) => s.id).map((s) => <a className="chip" key={s.id} href={`#${s.id}`}>{s.heading}</a>)}
          </nav>
        )}
        {headed.map((s, i) => (
          <section key={s.id ?? `intro-${i}`} id={s.id} className="stack-sm doc-section">
            {s.heading && <h2 className="h2">{s.heading}</h2>}
            {s.blocks.map((b, j) => b.list
              ? <ul key={j} className="policy-list">{b.list.map((item, k) => <li key={k} className="body-sm">{item}</li>)}</ul>
              : <p key={j} className="body-sm">{b.p}</p>)}
          </section>
        ))}
        <nav aria-label="Policies" className="cap policy-links">
          {POLICY_LINKS.filter((l) => l.type !== policy.policy_type).map((l, i) => (
            <span key={l.href}>{i > 0 && " · "}<Link href={l.href}>{l.label}</Link></span>
          ))}
        </nav>
        <p className="cap">Fingerprint of this text (SHA-256): <code className="hash">{policy.content_hash}</code></p>
      </article>
    </main><PublicFooter /></div>
  );
}
