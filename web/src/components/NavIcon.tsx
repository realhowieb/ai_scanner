/** Small line icons for the app navigation (24px grid, drawn with currentColor). Decorative:
 * every icon sits next to its text label, so they are hidden from screen readers. */
const PATHS: Record<string, string> = {
  today: "M3 11.5 12 4l9 7.5M5.5 9.5V20h13V9.5M10 20v-5h4v5",
  brief: "M4 5h12v14H6a2 2 0 0 1-2-2zM16 9h4v8a2 2 0 0 1-2 2h-2M7.5 9h5M7.5 12.5h5M7.5 16h3",
  scanner: "M4 6h16M4 12h11M4 18h7M18 15v6M15 18h6",
  daytrader: "M13 3 5 13.5h6L10 21l8-10.5h-6z",
  stocks: "M7 3.5v4M7 16.5v4M5 7.5h4v9H5zM17 3v4.5M17 15v6M15 7.5h4V15h-4z",
  watchlists: "m12 3.5 2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z",
  alerts: "M6 9a6 6 0 0 1 12 0c0 6.5 2.5 8 2.5 8h-17S6 15.5 6 9M10 20.5a2.2 2.2 0 0 0 4 0",
  track: "M4 4v16h16M7.5 15l3.5-4 3 2.5 5-6",
  ask: "M4 5h16v11h-9l-5 4v-4H4z",
  journal: "M6 3.5h11a2 2 0 0 1 2 2v15H7.5A2.5 2.5 0 0 1 5 18V4.5a1 1 0 0 1 1-1zM5 18a2.5 2.5 0 0 1 2.5-2.5H19M9 7.5h6",
  paper: "M3.5 7.5h17v12h-17zM16 13.5h2M3.5 7.5l2.5-3.5h12l2.5 3.5",
  more: "M5 12h.01M12 12h.01M19 12h.01",
};

export function NavIcon({ name, size = 18 }: { name: string; size?: number }) {
  const d = PATHS[name];
  if (!d) return null;
  return (
    <svg className="nav-icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={name === "more" ? 3 : 1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d={d} />
    </svg>
  );
}
