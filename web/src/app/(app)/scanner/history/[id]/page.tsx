"use client";

import { useParams } from "next/navigation";

import { SavedScanView } from "@/features/ScanHistoryView";

export default function SavedScanPage() {
  const { id } = useParams<{ id: string }>();
  const n = /^\d{1,12}$/.test(id || "") ? Number(id) : 0;
  return <SavedScanView id={n} />;
}
