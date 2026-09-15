// VoiceShield AI — Incident Log (privacy-safe: features only, no raw audio)

import { AttackTypeBadge } from './ComponentBreakdown';
import { SeverityDot } from '../icons';

function RiskBadge({ score, level }) {
  let cls = 'tag-safe';
  if (score >= 85 || level === 'CRITICAL') cls = 'tag-critical';
  else if (score >= 70 || level === 'HIGH') cls = 'tag-high';
  else if (score >= 30 || level === 'MEDIUM') cls = 'tag-medium';
  return (
    <span className={`tag ${cls}`}>
      <SeverityDot level={level || (score >= 85 ? 'CRITICAL' : score >= 70 ? 'HIGH' : score >= 30 ? 'MEDIUM' : 'LOW')} size={6} />
      {Math.round(score)}
    </span>
  );
}

function ActionBadge({ action }) {
  const map = {
    ALLOWED: { label: 'Allowed', color: '#2dd87b' },
    STEP_UP_VERIFICATION: { label: 'Step-Up', color: '#f5a623' },
    BLOCKED: { label: 'Blocked', color: '#ff6b3d' },
    ESCALATED: { label: 'Escalated', color: '#ff3355' },
  };
  const info = map[action] || { label: action, color: '#888' };
  return (
    <span className="action-badge" style={{ '--badge-color': info.color }}>
      {info.label}
    </span>
  );
}

export function IncidentLog({ incidents = [], isLoading = false }) {
  if (isLoading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {[1, 2, 3].map(i => (
          <div key={i} className="skeleton" style={{ height: 48, borderRadius: 8 }} />
        ))}
      </div>
    );
  }

  if (!incidents.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-title">No Incidents Recorded</div>
        <div className="empty-state-sub">Analyzed calls will appear here</div>
      </div>
    );
  }

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Session</th>
            <th>Timestamp</th>
            <th>Caller</th>
            <th>Risk</th>
            <th>Attack Type</th>
            <th>Action</th>
            <th>Transaction</th>
          </tr>
        </thead>
        <tbody>
          {incidents.map((inc, i) => (
            <tr key={i}>
              <td className="mono" title={inc.session_id}>
                {inc.session_id ? `${inc.session_id.slice(0, 8)}…` : '—'}
              </td>
              <td className="mono ts-cell">
                {inc.timestamp
                  ? new Date(inc.timestamp + 'Z').toLocaleString('en-IN', { hour12: true })
                  : '—'}
              </td>
              <td>{inc.caller_name || <span className="text-muted">Unknown</span>}</td>
              <td><RiskBadge score={inc.risk_score ?? 0} level={inc.risk_level} /></td>
              <td>
                {inc.attack_type
                  ? <AttackTypeBadge attackType={inc.attack_type} />
                  : <span className="text-muted">—</span>}
              </td>
              <td>
                {inc.action_taken
                  ? <ActionBadge action={inc.action_taken} />
                  : <span className="text-muted">—</span>}
              </td>
              <td>
                {inc.transaction_amount ? (
                  <span className="txn-cell">
                    <span className="mono">₹{Number(inc.transaction_amount).toLocaleString('en-IN')}</span>
                    <span className={inc.transaction_blocked ? 'txn-blocked' : 'txn-ok'}>
                      {inc.transaction_blocked ? 'BLOCKED' : 'CLEARED'}
                    </span>
                  </span>
                ) : <span className="text-muted">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="privacy-note">
        Privacy: Incident log stores acoustic feature metadata only. No raw audio is retained.
      </p>
    </div>
  );
}
