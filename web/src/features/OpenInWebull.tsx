// One-tap "Open in Webull": a plain link to /go/webull/{ticker}, which redirects to the
// stock's Webull page. Nothing is sent to Webull; HSF is research only.
import { captureUsage } from "@/lib/usageAnalytics";

export function webullHref(ticker: string): string {
  return `/go/webull/${encodeURIComponent(ticker.toUpperCase())}`;
}

export function OpenInWebull({ ticker, compact = false, small = false }: { ticker: string; compact?: boolean; small?: boolean }) {
  const t = ticker.toUpperCase();
  const track = () => captureUsage("webull_opened");
  if (compact) {
    return (
      <a className="icon-btn webull-link" href={webullHref(t)} target="_blank" rel="noopener noreferrer"
        aria-label={`Open ${t} in Webull`} title="Open in Webull" onClick={track}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
        </svg>
      </a>
    );
  }
  return (
    <a className={small ? "btn btn-sm webull-link" : "btn webull-link"} href={webullHref(t)} target="_blank" rel="noopener noreferrer"
      title={`Opens ${t} on webull.com (or the Webull app). HSF doesn't place trades.`} onClick={track}>
      Open in Webull
    </a>
  );
}
