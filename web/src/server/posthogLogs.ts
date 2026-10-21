import "server-only";

import { SeverityNumber } from "@opentelemetry/api-logs";
import { posthogLogProvider } from "../../instrumentation";

const logger = posthogLogProvider?.getLogger("posthog-exporter");

export function posthogLogsEnabled(): boolean { return !!logger; }

/** Emits only the bounded proxy lifecycle logs introduced by this integration. */
export function logProxyRequestStarted(method: string): void {
  try { logger?.emit({
    body: "HSF API proxy request started",
    severityNumber: SeverityNumber.INFO,
    attributes: { "http.request.method": method },
  }); } catch { /* Telemetry must not affect the proxy response. */ }
}

/** Records the proxy result without request paths, bodies, identities, or credentials. */
export function logProxyResponse(method: string, status: number): void {
  try { logger?.emit({
    body: status >= 500 ? "HSF API proxy request failed" : "HSF API proxy response sent",
    severityNumber: status >= 500 ? SeverityNumber.ERROR : SeverityNumber.INFO,
    attributes: {
      "http.request.method": method,
      "http.response.status_code": status,
    },
  }); } catch { /* Telemetry must not affect the proxy response. */ }
}

export async function flushPostHogLogs(): Promise<void> {
  try { await posthogLogProvider?.forceFlush(); } catch { /* Best effort. */ }
}
