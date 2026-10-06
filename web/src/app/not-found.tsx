import Link from "next/link";

export default function NotFound() {
  return (
    <main className="center">
      <div className="card narrow">
        <h1 className="h1">Page not found</h1>
        <p className="cap">That page doesn&apos;t exist.</p>
        <Link href="/today">Go to Today</Link>
      </div>
    </main>
  );
}
