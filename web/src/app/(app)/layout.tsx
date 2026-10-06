import type { ReactNode } from "react";

import { Protected } from "@/components/Protected";

export default function AppLayout({ children }: { children: ReactNode }) {
  return <Protected>{children}</Protected>;
}
