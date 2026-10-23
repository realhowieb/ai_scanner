/**
 * "What's new" card (components/WhatsNew.tsx). Newest first. Add an entry to the top in
 * any PR that ships something a customer would notice, in plain words; the card shows
 * the newest entry once per browser after the promote that deploys it. `id` must be
 * unique and only ever grow (a date works).
 */
export type WhatsNewEntry = { id: string; title: string; items: string[] };

export const WHATS_NEW: WhatsNewEntry[] = [
  {
    id: "2026-10-23",
    title: "What's new",
    items: [
      "A weekly recap email on Saturdays: how last week's top Scanner picks did, your watchlist's movers and the week ahead.",
      "Faster Today and Scanner pages when you come back to them.",
      "The Earnings scanner scores every company reporting this week.",
      "Telegram and Discord alerts, and SEC insider-buy alerts for Pro.",
    ],
  },
];
