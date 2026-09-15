// VoiceShield AI — Security Alert Panel

import { SeverityDot } from '../icons';

function getAlertClass(level) {
  const map = { CRITICAL: 'alert-critical', HIGH: 'alert-high', MEDIUM: 'alert-medium' };
  return map[level] || '';
}

export function AlertPanel({ alerts = [], onAcknowledge }) {
  if (!alerts.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon"><SeverityDot level="LOW" size={16} /></div>
        <div className="empty-state-title">No Active Alerts</div>
        <div className="empty-state-sub">All monitored channels are clear</div>
      </div>
    );
  }

  return (
    <div>
      {alerts.map((alert, i) => (
        <div key={i} className={`alert-item ${getAlertClass(alert.level)}`}>
          <SeverityDot level={alert.level} size={10} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="alert-title">{alert.title}</div>
            <div className="alert-message">{alert.message}</div>
            {alert.actions?.length > 0 && (
              <div className="alert-actions">
                {alert.actions.map((action, j) => (
                  <span key={j} className="alert-action-chip">{action}</span>
                ))}
              </div>
            )}
            <div className="alert-time">{alert.time}</div>
            {alert.channels?.length > 0 && (
              <div className="channel-tags">
                {alert.channels.map((ch, k) => (
                  <span key={k} className={`channel-tag channel-${ch.channel}`}>
                    {ch.channel.toUpperCase()}: {ch.status}
                  </span>
                ))}
              </div>
            )}
          </div>
          {onAcknowledge && (
            <button className="btn btn-secondary btn-sm" onClick={() => onAcknowledge(i)}>
              Acknowledge
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

export function ActionCard({ decision, verifications = [] }) {
  if (!decision) return null;

  const isBlocked = decision === 'BLOCKED' || decision === 'ESCALATED';
  const isStepUp = decision === 'STEP_UP_VERIFICATION';

  return (
    <div className={`action-card ${isBlocked ? 'blocked' : isStepUp ? 'stepup' : 'allowed'}`}>
      <div className="action-card-title">
        {isBlocked ? 'ACTION BLOCKED' : isStepUp ? 'STEP-UP VERIFICATION REQUIRED' : 'ALLOWED'}
      </div>
      {verifications.length > 0 && (
        <div>
          <div className="action-card-subtitle">Required Verifications</div>
          <div className="action-card-list">
            {verifications.map((v, i) => (
              <div key={i} className="action-card-item">{v}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
