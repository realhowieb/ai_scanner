import Link from "@/components/Link";

// Signed-out visitors land here too, so "home" (which sends members on to Today) beats a
// signed-in-only page.
export default function NotFound() {
  return (
    <main className="center">
      <div className="card narrow login stack">
        <Link href="/" className="brand brand-lg" aria-label="HSFinest.AI home">HSFinest<span>.AI</span></Link>
        <h1 className="h1">Page not found</h1>
        <p className="cap">That page doesn&apos;t exist, or the link is out of date.</p>
        <div className="row-actions">
          <Link href="/" className="btn btn-primary">Go to home</Link>
          <Link href="/pricing">Pricing</Link>
          <Link href="/how-hsf-works">How HSF works</Link>
        </div>
      </div>
    </main>
  );
}
