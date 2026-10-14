import type { Metadata } from "next";

import { AdminView } from "@/features/AdminView";

export const metadata: Metadata = { title: "Admin" };

export default function Page() {
  return <AdminView />;
}
