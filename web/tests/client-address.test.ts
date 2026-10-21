// @vitest-environment node
import { describe, expect, it } from "vitest";

import { CLIENT_IP_HEADER, PROXY_KEY_HEADER, clientAddress, withClientAddress } from "@/server/clientAddress";

const req = (xff?: string) => new Request("https://web.test/api/auth/login", { headers: xff ? { "x-forwarded-for": xff } : {} });

describe("visitor address for the API's per-address limits", () => {
  it("takes the last X-Forwarded-For entry (the one Render's proxy added)", () => {
    expect(clientAddress(req("6.6.6.6, 203.0.113.9"))).toBe("203.0.113.9");
    expect(clientAddress(req("2001:db8::1"))).toBe("2001:db8::1");
    expect(clientAddress(req("not an ip"))).toBeNull();
    expect(clientAddress(req())).toBeNull();
  });

  it("adds the address and secret only when a secret is set", async () => {
    const seen: Headers[] = [];
    const up = async (_p: string, init: RequestInit) => { seen.push(new Headers(init.headers)); return new Response("{}"); };
    await withClientAddress(req("203.0.113.9"), up, "s3cret")("/v1/auth/login", { method: "POST", headers: { "x-request-id": "abc12345" } });
    expect(seen[0]!.get(CLIENT_IP_HEADER)).toBe("203.0.113.9");
    expect(seen[0]!.get(PROXY_KEY_HEADER)).toBe("s3cret");
    expect(seen[0]!.get("x-request-id")).toBe("abc12345");
    await withClientAddress(req("203.0.113.9"), up, "")("/v1/auth/login", { method: "POST" });
    expect(seen[1]!.get(CLIENT_IP_HEADER)).toBeNull();
    expect(seen[1]!.get(PROXY_KEY_HEADER)).toBeNull();
  });
});
