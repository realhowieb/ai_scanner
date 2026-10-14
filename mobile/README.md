# HSF iPhone app

A [Capacitor](https://capacitorjs.com) shell that opens the live web app
(`https://hsf-web-beta.onrender.com/today`, see `capacitor.config.ts`). Web releases
reach the app right away; only native changes (icon, push, plugins) need a new build.

- `capacitor.config.ts`: app id `com.hsfinest.app`, name "HSF", the site it opens.
- `ios/`: the Xcode project (Swift Package Manager, no CocoaPods). iPhone only, portrait.
- `www/index.html`: shown only if the site can't load at all.
- Builds run on GitHub's Mac runners (`.github/workflows/ios.yml`); no Mac needed.

## One-time setup (account holder)

1. **Find your Team ID.** developer.apple.com → Account → Membership details → Team ID.
2. **Create the app record.** appstoreconnect.apple.com → Apps → + → New App:
   platform iOS, name "HSF" (or "HSFinest.AI" if taken), language English,
   bundle ID `com.hsfinest.app` (if it isn't listed, register it first at
   developer.apple.com → Certificates, IDs & Profiles → Identifiers → +, App IDs,
   with **Push Notifications** ticked), SKU `hsf-ios`.
3. **Create an API key.** App Store Connect → Users and Access → Integrations →
   App Store Connect API → Team Keys → +. Name "GitHub builds", access **Admin**
   (needed so builds can create the signing certificate and profile). Download the
   `.p8` file (it downloads once) and note the Key ID and the Issuer ID above the table.
4. **Add four GitHub secrets.** github.com/realhowieb/ai_scanner → Settings →
   Secrets and variables → Actions → New repository secret:
   - `APPLE_TEAM_ID`: the Team ID
   - `ASC_KEY_ID`: the Key ID
   - `ASC_ISSUER_ID`: the Issuer ID
   - `ASC_KEY_P8`: the whole contents of the `.p8` file, including the BEGIN/END lines

## Making a build

Actions → **iOS app** → Run workflow (branch `main`). It archives, signs and uploads to
App Store Connect. After Apple processes it, it appears under TestFlight; install the
TestFlight app on your iPhone to try it. The build number is the workflow run number.

Pull requests that change `mobile/` only build for the simulator, with no secrets.

## Local work (optional, needs a Mac with Xcode)

```sh
cd mobile && npm ci && npx cap sync ios && open ios/App/App.xcodeproj
```

`npx cap sync ios` must run after `npm ci` or a config change: it writes the
generated `capacitor.config.json` and `public/` into the Xcode project (both gitignored).

## Push alerts (hsf-api)

Alerts reach the app through Apple Push Notification service (`api/apns.py`). On
developer.apple.com → Certificates, IDs & Profiles → **Keys** → +, tick "Apple Push
Notifications service (APNs)", download the `.p8`, and set on hsf-api (Render → Environment):

- `APNS_KEY_P8`: the `.p8` contents (literal `\n` line breaks are fine)
- `APNS_KEY_ID`: that key's ID
- `APNS_TEAM_ID`: the Team ID

Then the app's Alerts page shows "Notify me on this iPhone" (Pro and above).
`HSF_APNS_ENABLED=0` turns it off without removing the key; `APNS_SANDBOX=1` is only
for debug builds run from Xcode.
