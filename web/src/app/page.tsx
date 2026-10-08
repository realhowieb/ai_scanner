import type { Metadata } from "next";

import { Landing } from "@/features/Landing";

export const metadata: Metadata = {
  title: { absolute: "HSF AI Stock Scanner · HSFinest.AI" },
  description: "HSF continuously scans the U.S. stock market and helps traders identify, understand and monitor noteworthy technical setups.",
};

// Signed-in visitors never see this: the proxy sends "/" to /today for them.
export default function Home() {
  return <Landing />;
}
