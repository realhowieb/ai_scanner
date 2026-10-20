"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { startNativePush } from "@/lib/nativePush";

/** Inside the HSF iPhone app only: keeps its push token current and opens tapped
 * notifications. Renders nothing; does nothing in a browser. */
export function NativeApp() {
  const router = useRouter();
  useEffect(() => { void startNativePush((url) => router.push(url)); }, [router]);
  return null;
}
