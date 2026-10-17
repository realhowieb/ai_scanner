import Link from "next/link";

import { Pill, ScoreBadge } from "@/components/ui";
import { SAMPLE_SETUPS } from "@/lib/sampleResearch";

export function SampleSetupPreview({ href = "/demo#demo-stock", onReview }: { href?: string; onReview?: () => void }) {
  const sample = SAMPLE_SETUPS[0]!;
  return <aside className="card sample-preview" aria-label="Sample ranked setup">
    <div className="row"><span className="eyebrow">Sample data · not live</span></div>
    <div className="row"><strong className="tk">{sample.ticker}</strong><Pill>{sample.setup}</Pill><span className="grow" /><ScoreBadge score={sample.score} /></div>
    <dl className="sample-facts"><div><dt>Sample price</dt><dd>{sample.last}</dd></div><div><dt>Relative volume</dt><dd>{sample.rvol}</dd></div></dl>
    <p className="body-sm"><strong>Why it ranks:</strong> {sample.note}</p>
    <p className="cap"><strong>Risk to review:</strong> {sample.risk}</p>
    <Link href={href} onClick={onReview} className="sample-review-link">Explore the sample evidence →</Link>
  </aside>;
}
