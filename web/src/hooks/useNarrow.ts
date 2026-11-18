"use client";
import { useEffect, useState } from "react";

/** Phone layout (compact rows, no table) at or below this width. */
export function useNarrow(maxWidth = 760): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia(`(max-width: ${maxWidth}px)`);
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, [maxWidth]);
  return narrow;
}
