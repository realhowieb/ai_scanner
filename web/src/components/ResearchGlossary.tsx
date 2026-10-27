export function ResearchGlossary() {
  return <details className="research-glossary"><summary>Score and volume definitions</summary>
    <dl className="kv">
      <div><dt>HSF Score</dt><dd>0–100 ranking of current technical evidence, not a probability of profit.</dd></div>
      <div><dt>DT score</dt><dd>Intraday structure ranking for Day Trader; separate from HSF Score.</dd></div>
      <div><dt>RVOL</dt><dd>Volume relative to its recent average. 2× means twice that average.</dd></div>
      <div><dt>PreBreakout</dt><dd>Model estimate for a defined setup event. “Top %” is its rank within the scan, not a win rate.</dd></div>
    </dl>
  </details>;
}
