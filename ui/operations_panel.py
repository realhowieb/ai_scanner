"""Escaped rendering of the same allowlisted operations summary as the API."""
import html


def operations_html(report):
    def esc(value):
        return html.escape(str(value)) if value is not None else 'Unavailable'
    mat = report['maturation']
    refresh = ' · refresh failed; showing retained snapshot' if report.get('refresh_failed') else ''
    rows = ''.join('<tr>' + ''.join(f'<td>{esc(row.get(key))}</td>' for key in
                   ('label', 'status', 'last_success', 'last_failure', 'consecutive_failures')) + '</tr>'
                   for row in report['workflows'])
    traffic = report['database_traffic']
    totals = traffic['sample_totals']
    estimates = ''.join(f"<tr><td>{esc(row['workflow'])}</td>"
                        f"<td>{esc(row['metrics']['db_to_client_payload_bytes'])}</td>"
                        f"<td>{esc(row['metrics']['client_to_db_payload_bytes'])}</td>"
                        f"<td>{esc(row['metrics']['calls'])}</td></tr>" for row in traffic['workflows'])
    return ('<section class="hsf-operations" aria-label="Processing health"><h3>Processing health</h3>'
            f"<p>{'Stale' if report['stale'] else 'Current'} snapshot · {esc(report['generated_at'])}{refresh}</p>"
            f"<p>Last successful maturation: {esc(mat['last_success_at'])}<br>"
            f"Pending ready observations: {esc(mat['pending_observations'])} · "
            f"Oldest pending age: {esc(mat['oldest_pending_age_minutes'])} minutes</p>"
            f"<p><small>{esc(mat['scope'])}</small></p>"
            '<table><thead><tr><th>Workflow</th><th>Status</th><th>Last success</th>'
            '<th>Last failure</th><th>Consecutive failures</th></tr></thead>'
            f'<tbody>{rows}</tbody></table><h4>Estimated database traffic</h4>'
            '<p>Application payload estimates, not Neon billed usage.</p>'
            f"<p>Sample total · database → client: {esc(totals['db_to_client_payload_bytes'])} bytes · "
            f"client → database: {esc(totals['client_to_db_payload_bytes'])} bytes</p>"
            f"<p><small>{esc(traffic['scope'])}</small></p>"
            '<table><thead><tr><th>Workflow</th><th>Database → client bytes</th>'
            f'<th>Client → database bytes</th><th>Calls</th></tr></thead><tbody>{estimates}</tbody></table>'
            f"<p>{'' if traffic['available'] else 'Traffic telemetry unavailable.'}</p>"
            '<p>Neon provider usage: Unavailable</p></section>')


def render_operations_panel(summary):
    import streamlit as st
    if not bool(st.session_state.get('is_admin')):
        return
    st.markdown(operations_html(summary), unsafe_allow_html=True)
