# www.hsfinest.com: the app's front door

`www.hsfinest.com` (and `hsfinest.com`) point at the same Render service as the app
(`hsf-web-v1`). On those hosts the web app (`web/src/proxy.ts`, `web/src/lib/siteHosts.ts`):

- serves the public pages: `/`, `/pricing`, `/how-hsf-works`, `/demo`, `/weekly`, `/founders`,
  `/founding-100/*`, `/terms`, `/privacy`, `/share/*`, plus the public API calls and files they use;
- sends every other path (sign in, sign up, the app) to `https://app.hsfinest.com` with the same
  path and query (307), so sessions live on one host;
- sends the old music site's pages (`/shop`, `/blog-hsf`, `/music`, `/events`, `/services`) to
  `https://music.hsfinest.com` (301).

Links clicked on www (Sign in, Create free account) go straight to the app with the visitor's
utm tags, invite code and outside referrer (`hsf_referrer`), so sign-ups keep their source.

## Cutover (in this order)

1. Hostinger Website Builder: connect the music site to the subdomain `music.hsfinest.com`
   (otherwise it goes offline in step 3).
2. Render > hsf-web-v1 > Settings > Custom Domains: add `www.hsfinest.com` (Render adds
   `hsfinest.com` too and redirects it to www). Render shows the DNS records to create.
3. Hostinger DNS for hsfinest.com: `CNAME www -> hsf-web-v1.onrender.com` and the `A @` record
   Render shows for the bare domain. Remove the old www/@ records that point at the builder.
4. Wait for Render to verify both domains and issue certificates.
5. Set `NEXT_PUBLIC_SITE_URL=https://www.hsfinest.com` on hsf-web-v1 (rebuilds). Canonical
   links, the sitemap and link previews then use www.

Leave `APP_BASE_URL` on hsf-api-v1 at `https://app.hsfinest.com`: emails and Stripe returns
should keep landing in the app.
