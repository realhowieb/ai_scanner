// Renders the AI's Markdown as plain React elements (no HTML injection): paragraphs,
// headings, bullet and numbered lists, and **bold**. Anything else shows as text.
import type { ReactNode } from "react";

function inline(text: string, key: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4
      ? <strong key={`${key}-${i}`}>{part.slice(2, -2)}</strong>
      : part.replace(/(^|\s)[*_]([^*_\s][^*_]*)[*_](?=\s|$|[.,;:])/g, "$1$2"));
}

export function AIText({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (!list) return;
    const items = list.items.map((t, i) => <li key={i}>{inline(t, `li${blocks.length}-${i}`)}</li>);
    blocks.push(list.ordered ? <ol key={blocks.length}>{items}</ol> : <ul key={blocks.length}>{items}</ul>);
    list = null;
  };
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const ordered = !!numbered;
      if (list && list.ordered !== ordered) flush();
      list ??= { ordered, items: [] };
      list.items.push((bullet ?? numbered)![1]!);
      continue;
    }
    flush();
    if (!line) continue;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    blocks.push(heading
      ? <p key={blocks.length} className="strong">{inline(heading[1]!.replace(/\*\*/g, ""), `h${blocks.length}`)}</p>
      : <p key={blocks.length}>{inline(line, `p${blocks.length}`)}</p>);
  }
  flush();
  return <div className="ai-md body-sm">{blocks}</div>;
}
