import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { BatchLogRecordProcessor, LoggerProvider } from "@opentelemetry/sdk-logs";
import { posthogHost } from "./src/lib/posthogConfig";

function createLogProvider(): LoggerProvider | null {
  const projectKey = process.env.NEXT_PUBLIC_POSTHOG_KEY?.trim();
  if (process.env.POSTHOG_LOGS_ENABLED !== "1" || !projectKey) return null;
  try {
    return new LoggerProvider({
      resource: resourceFromAttributes({ "service.name": "hsf-web" }),
      processors: [new BatchLogRecordProcessor({
        exporter: new OTLPLogExporter({
          url: new URL("/i/v1/logs", posthogHost(process.env.NEXT_PUBLIC_POSTHOG_HOST)).toString(),
          headers: { Authorization: `Bearer ${projectKey}`, "Content-Type": "application/json" },
        }),
      })],
    });
  } catch { return null; }
}

/** Explicitly enabled operational logs; never register a global log exporter. */
export const posthogLogProvider = createLogProvider();
export function register(): void {}
