import type { CapacitorConfig } from "@capacitor/cli";

// The app opens the live web app, so every web release reaches the iPhone app without a
// new App Store build. Only native changes (push, icons, plugins) need a TestFlight build.
// HSF_APP_URL overrides the site for a test build, e.g. a dev deploy.
const appUrl = process.env.HSF_APP_URL || "https://hsf-web-v1.onrender.com";

const config: CapacitorConfig = {
  appId: "com.hsfinest.app",
  appName: "HSF",
  webDir: "www", // shown only if the site can't load at all
  backgroundColor: "#0b0f14",
  server: {
    url: `${appUrl}/today`,
    cleartext: false,
  },
  ios: {
    contentInset: "always", // keep pages clear of the notch and home bar
    backgroundColor: "#0b0f14",
    limitsNavigationsToAppBoundDomains: false,
  },
  plugins: {
    PushNotifications: { presentationOptions: ["badge", "sound", "alert"] },
  },
};

export default config;
