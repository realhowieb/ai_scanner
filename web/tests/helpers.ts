export function jwt(expInS: number, nowMs = Date.now()): string {
  const enc = (o: object) => Buffer.from(JSON.stringify(o)).toString("base64url");
  return `${enc({ alg: "HS256" })}.${enc({ sub: "u", exp: Math.floor(nowMs / 1000) + expInS })}.sig`;
}

export function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(body === null ? null : JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

export function setCookies(res: Response): string[] {
  return res.headers.getSetCookie();
}
