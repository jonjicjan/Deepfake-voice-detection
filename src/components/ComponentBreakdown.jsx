// VoiceShield AI — Component Score Breakdown

const COMPONENTS = [
  { key: 'deepfake_probability', label: 'Deepfake Detection', desc: 'Neural TTS / synthesis artifacts' },
  { key: 'speaker_mismatch', label: 'Speaker Mismatch', desc: 'Identity vs enrolled profile' },
  { key: 'prosody_anomaly', label: 'Prosody Anomaly', desc: 'Pitch, rhythm, pause patterns' },
  { key: 'replay_probability', label: 'Replay Attack', desc: 'Pre-recorded audio detection' },
  { key: 'unknown_attack_score', label: 'Unknown Synthetic', desc: 'Unseen attack patterns' },
  { key: 'context_risk', label: 'Context Risk', desc: 'Transaction and caller context' },
];

function getScoreColor(value) {
  if (value >= 0.7) return { fill: '#ff3355', text: '#ff3355' };
  if (value >= 0.5) return { fill: '#ff6b3d', text: '#ff6b3d' };
  if (value >= 0.3) return { fill: '#f5a623', text: '#f5a623' };
  return { fill: '#2dd87b', text: '#2dd87b' };
}

export function ComponentBreakdown({ scores }) {
  if (!scores) {
    return (
      <div className="score-bar-row">
        {COMPONENTS.map(c => (
          <div key={c.key} className="score-item">
            <div className="score-item-header">
              <span className="score-item-label">{c.label}</span>
              <span className="score-item-value text-muted">—</span>
            </div>
            <div className="score-track">
              <div className="score-fill skeleton" style={{ width: '100%', height: '100%' }} />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="score-bar-row">
      {COMPONENTS.map(c => {
        const value = scores[c.key] ?? 0;
        const pct = Math.round(value * 100);
        const color = getScoreColor(value);
        return (
          <div key={c.key} className="score-item">
            <div className="score-item-header">
              <span className="score-item-label" title={c.desc}>{c.label}</span>
              <span className="score-item-value" style={{ color: color.text }}>{pct}%</span>
            </div>
            <div className="score-track">
              <div
                className="score-fill"
                style={{
                  width: `${pct}%`,
                  background: `linear-gradient(90deg, ${color.fill}99, ${color.fill})`,
                  boxShadow: pct >= 50 ? `0 0 8px ${color.fill}55` : 'none',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function AttackTypeBadge({ attackType }) {
  const attackMap = {
    TTS_SYNTHESIZED: { label: 'TTS Synthesized', cls: 'attack-TTS' },
    VOICE_CONVERSION: { label: 'Voice Conversion', cls: 'attack-VOICE_CONVERSION' },
    REPLAY_ATTACK: { label: 'Replay Attack', cls: 'attack-REPLAY' },
    UNKNOWN_SYNTHETIC: { label: 'Unknown Synthetic', cls: 'attack-UNKNOWN' },
    GENUINE: { label: 'Genuine Voice', cls: 'attack-GENUINE' },
  };
  const info = attackMap[attackType] || { label: attackType, cls: 'tag-info' };
  return <span className={`tag ${info.cls}`}>{info.label}</span>;
}
