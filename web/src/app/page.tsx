import { redirect } from "next/navigation";

// The proxy sends "/" to /today or /login; this is the fallback.
export default function Home() {
  redirect("/today");
}
