import Link from "@/components/Link";

/** Terms and privacy for the public and signed-in footers, plus the Refund Policy while the
 * Founding 100 is open (it only covers founding memberships and isn't published otherwise). */
export function LegalLinks({ refunds = false }: { refunds?: boolean }) {
  return <><Link href="/terms">Terms</Link> · <Link href="/privacy">Privacy</Link>{refunds && <> · <Link href="/founding-100/refunds">Refund Policy</Link></>}</>;
}
