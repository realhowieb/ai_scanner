"""Render real UI helpers with clearly synthetic data; no database/network IO."""
import datetime as dt
from pathlib import Path

from analytics.operations import summary
from ui.operations_panel import operations_html
from ui.trust_banner import banner_html, build_trust_info


def preview():
    now = dt.datetime(2026, 10, 9, 21, tzinfo=dt.timezone.utc)
    report = {'generated_at': now.isoformat(), 'system_status': 'HEALTHY',
              'subsystems': {name: {'status': 'HEALTHY'} for name in ('universe', 'scanner', 'market_data', 'database')}}
    report['subsystems']['maturation'] = {'metrics': {'last_success': now.isoformat(), 'ready_observations': 12,
                                                    'oldest_pending_age_min': 35}}
    report['subsystems']['workflows'] = {'metrics': {'workflows': [{
        'workflow': 'scheduled-scans.yml', 'status': 'HEALTHY', 'last_success': now.isoformat(),
        'last_failure': None, 'consecutive_failures': 0}]}}
    report['database_traffic'] = [{'workflow': 'scheduled-scans.yml', 'observed_at': now.isoformat(),
                                  'metrics': {'measurement': 'application_payload_estimate', 'calls': 1631,
                                              'db_to_client_payload_bytes': 13520963,
                                              'client_to_db_payload_bytes': 2146392}}]
    recent = [{'username': 'cron', 'label': 'US_MARKET', 'created_at': now, 'row_count': 100}]
    stale = [{**recent[0], 'created_at': now - dt.timedelta(days=1)}]
    banners = ''.join(banner_html(build_trust_info(runs, report, now)) for runs in (recent, stale, []))
    panel = operations_html(summary(report, now=now))
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">' + '''
    <title>HSF freshness and processing health preview</title>
    <style>body{background:#101821;color:#e4edf4;font:16px system-ui;margin:0;padding:32px}
    main{max-width:1080px;margin:auto}h1{font-size:28px}h3{font-size:22px}p{line-height:1.6}
    small{color:#a6b4c4}section{background:#182430;border:1px solid #304253;border-radius:12px;padding:24px;margin-top:28px;overflow:auto}
    table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:12px 10px;border-bottom:1px solid #304253}
    th{color:#a8bdd1}h4{margin-top:28px}.hsf-trust{background:#182430;margin-bottom:14px!important;padding:16px!important}
    .demo{color:#a6b4c4}@media(max-width:640px){body{padding:16px}section{padding:16px}}</style>
    <main><h1>HSF · Data freshness & processing health</h1>
    <p class="demo">UI preview · synthetic examples, not production metrics.<br>Public freshness banner states, followed by the admin-only processing panel.</p>
    ''' + banners + panel + '</main></html>'


if __name__ == '__main__':
    destination = Path('docs/previews/freshness-operations.html')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(preview())
    print(destination)
