// VoiceShield AI — Cyber Security Operations Console

import { useState, useEffect, useRef, useCallback } from 'react';
import { WaveformVisualizer } from './components/WaveformVisualizer';
import { RiskMeter } from './components/RiskMeter';
import { ComponentBreakdown, AttackTypeBadge } from './components/ComponentBreakdown';
import { AlertPanel, ActionCard } from './components/AlertPanel';
import { IncidentLog } from './components/IncidentLog';
import { LiveMicMonitor } from './components/LiveMicMonitor';
import { apiFetch, getApiKey, setApiKey, clearApiKey, checkHealth } from './api';
import {
  IconShield, IconMic, IconUpload, IconChart, IconList,
  IconSettings, IconActivity, IconLogout, SeverityDot,
} from './icons';

const PAGE_TITLES = {
  live: 'Voice Analysis',
  incidents: 'Incident Log',
  policy: 'Policy Configuration',
  system: 'System Diagnostics',
};

function getLevelColors(level) {
  return {
    LOW: { primary: '#2dd87b', dim: 'rgba(45,216,123,0.15)', text: '#2dd87b' },
    MEDIUM: { primary: '#f5a623', dim: 'rgba(245,166,35,0.15)', text: '#f5a623' },
    HIGH: { primary: '#ff6b3d', dim: 'rgba(255,107,61,0.15)', text: '#ff6b3d' },
    CRITICAL: { primary: '#ff3355', dim: 'rgba(255,51,85,0.15)', text: '#ff3355' },
  }[level] || { primary: '#4a9eff', dim: 'rgba(74,158,255,0.15)', text: '#4a9eff' };
}

function StatCard({ value, label, sublabel, color }) {
  return (
    <div className="stat-card">
      <div className="stat-value" style={{ color }}>{value}</div>
      <div className="stat-label">{label}</div>
      {sublabel && <div className="stat-delta text-muted">{sublabel}</div>}
    </div>
  );
}

function LoginPage({ onLogin }) {
  const [key, setKey] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setApiKey(key.trim());
    try {
      const res = await apiFetch('/stats');
      if (!res.ok) throw new Error('Invalid API key');
      onLogin();
    } catch {
      clearApiKey();
      setError('Authentication failed. Verify your API key and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">
          <IconShield size={36} />
          <div>
            <h1>VoiceShield AI</h1>
            <p>Voice Integrity Verification Platform</p>
          </div>
        </div>
        <form onSubmit={handleSubmit}>
          <label className="form-label" htmlFor="api-key">Operator API Key</label>
          <input
            id="api-key"
            className="form-input"
            type="password"
            value={key}
            onChange={e => setKey(e.target.value)}
            placeholder="Enter your API key"
            autoComplete="off"
            required
          />
          {error && <div className="form-error">{error}</div>}
          <button className="btn btn-primary w-full" type="submit" disabled={loading}>
            {loading ? 'Authenticating…' : 'Sign In'}
          </button>
        </form>
        <p className="login-footer">Authorized personnel only. All access is logged.</p>
      </div>
    </div>
  );
}

function CallContextForm({ context, onChange }) {
  return (
    <div className="context-form">
      <div className="form-row">
        <div className="form-group">
          <label className="form-label">Caller ID</label>
          <input className="form-input" value={context.callerId}
            onChange={e => onChange({ ...context, callerId: e.target.value })}
            placeholder="Employee / account ID" />
        </div>
        <div className="form-group">
          <label className="form-label">Caller Name</label>
          <input className="form-input" value={context.callerName}
            onChange={e => onChange({ ...context, callerName: e.target.value })}
            placeholder="Full name" />
        </div>
      </div>
      <div className="form-row">
        <div className="form-group">
          <label className="form-label">Phone Number</label>
          <input className="form-input" value={context.phone}
            onChange={e => onChange({ ...context, phone: e.target.value })}
            placeholder="+91-XXXXXXXXXX" />
        </div>
        <div className="form-group">
          <label className="form-label">Call Origin</label>
          <select className="form-input" value={context.origin}
            onChange={e => onChange({ ...context, origin: e.target.value })}>
            <option value="UNKNOWN">Unknown</option>
            <option value="MOBILE">Mobile</option>
            <option value="LANDLINE">Landline</option>
            <option value="VOIP">VoIP</option>
          </select>
        </div>
      </div>
      <div className="form-row">
        <div className="form-group">
          <label className="form-label">Transaction Amount (INR)</label>
          <input className="form-input" type="number" min="0" value={context.amount}
            onChange={e => onChange({ ...context, amount: e.target.value })}
            placeholder="0" />
        </div>
        <div className="form-group">
          <label className="form-label">Transaction Type</label>
          <select className="form-input" value={context.txnType}
            onChange={e => onChange({ ...context, txnType: e.target.value })}>
            <option value="">None</option>
            <option value="FUND_TRANSFER">Fund Transfer</option>
            <option value="WIRE_TRANSFER">Wire Transfer</option>
            <option value="APPROVAL">Approval Request</option>
            <option value="DATA_ACCESS">Data Access</option>
          </select>
        </div>
      </div>
      <div className="form-checks">
        <label className="check-label">
          <input type="checkbox" checked={context.isKnown}
            onChange={e => onChange({ ...context, isKnown: e.target.checked })} />
          Known contact
        </label>
        <label className="check-label">
          <input type="checkbox" checked={context.privileged}
            onChange={e => onChange({ ...context, privileged: e.target.checked })} />
          Privileged workflow
        </label>
      </div>
    </div>
  );
}

function LiveAnalysisPage({ stats, onNewIncident }) {
  const [file, setFile] = useState(null);
  const [context, setContext] = useState({
    callerId: '', callerName: '', phone: '', origin: 'UNKNOWN',
    amount: '', txnType: '', isKnown: false, privileged: false,
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const [liveScore, setLiveScore] = useState(0);
  const [liveLevel, setLiveLevel] = useState('LOW');
  const [liveMicActive, setLiveMicActive] = useState(false);
  const fileInputRef = useRef(null);

  const animateToScore = useCallback((targetScore, targetLevel, duration = 1200) => {
    const start = liveScore;
    const startTime = Date.now();
    const tick = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setLiveScore(start + (targetScore - start) * eased);
      if (progress < 1) requestAnimationFrame(tick);
      else { setLiveScore(targetScore); setLiveLevel(targetLevel); }
    };
    requestAnimationFrame(tick);
  }, [liveScore]);

  const applyAnalysisResult = useCallback((data) => {
    setResult(data);
    animateToScore(data.risk_score, data.risk_level);
    setAlerts(buildAlerts(data));
    onNewIncident({
      session_id: data.session_id,
      timestamp: new Date().toISOString().slice(0, -1),
      caller_name: context.callerName || data.caller_name,
      risk_score: data.risk_score,
      risk_level: data.risk_level,
      attack_type: data.attack_type,
      action_taken: data.action_recommended,
      transaction_amount: context.amount ? Number(context.amount) : null,
      transaction_blocked: data.transaction_blocked ? 1 : 0,
    });
  }, [animateToScore, onNewIncident, context]);

  const buildFormData = () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('call_origin', context.origin);
    if (context.callerId) formData.append('caller_id', context.callerId);
    if (context.callerName) formData.append('caller_name', context.callerName);
    if (context.phone) formData.append('phone_number', context.phone);
    formData.append('is_known_contact', String(context.isKnown));
    if (context.amount) {
      formData.append('transaction_amount', context.amount);
      formData.append('transaction_type', context.txnType || 'FUND_TRANSFER');
      formData.append('is_privileged_workflow', String(context.privileged));
    }
    return formData;
  };

  const runAnalysis = async () => {
    if (!file) return;
    setResult(null);
    setError(null);
    setIsAnalyzing(true);
    setIsLive(true);
    setAlerts([]);
    setLiveScore(0);
    setLiveLevel('LOW');
    try {
      const res = await apiFetch('/analyze-audio', { method: 'POST', body: buildFormData() });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }
      applyAnalysisResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleLiveRiskUpdate = useCallback((data) => {
    setIsLive(true);
    animateToScore(data.risk_score, data.risk_level);
    const liveResult = {
      session_id: data.session_id,
      risk_score: data.risk_score,
      risk_level: data.risk_level,
      attack_type: data.attack_type,
      component_scores: data.component_scores,
      action_recommended: data.action_recommended,
      alert_message: data.alert_message,
      explanation: data.explanation,
      transaction_blocked: data.transaction_blocked,
      processing_time_ms: null,
      audio_duration_s: 3.0,
      models_used: data.models_used,
      fraud_registry_match: data.fraud_registry_match,
      fraud_registry_detail: data.fraud_registry_detail,
      cross_session: data.cross_session,
      channel_notifications: data.channel_notifications,
    };
    setResult(liveResult);
    setAlerts(buildAlerts(liveResult));
  }, [animateToScore]);

  const colors = getLevelColors(liveLevel);

  return (
    <div>
      <div className="stats-grid stats-grid-4">
        <StatCard value={stats.total} label="Calls Analyzed" sublabel="All time" color="#4a9eff" />
        <StatCard value={stats.highRisk} label="High Risk" sublabel="Score ≥ 70" color="#ff3355" />
        <StatCard value={stats.blocked} label="Blocked" sublabel="Transactions" color="#ff6b3d" />
        <StatCard value={stats.avgScore} label="Avg Risk Score" sublabel="Lower is safer" color="#f5a623" />
      </div>

      <div className="analysis-grid">
        <div className="analysis-left">
          <div className="card" style={{
            borderColor: result ? `${colors.primary}40` : undefined,
            boxShadow: result ? `0 0 30px ${colors.primary}18` : undefined,
          }}>
            <div className="card-header">
              <div className="card-title">Impersonation Risk Score</div>
              {isAnalyzing && <div className="spinner" />}
            </div>
            <WaveformVisualizer isActive={isLive} riskLevel={liveLevel} barCount={40} />
            <div className="risk-meter-wrap">
              <RiskMeter score={liveScore} level={liveLevel} size={200} />
            </div>
            {result && (
              <div className="metrics-row">
                <div className="metric">
                  <div className="metric-value">{result.processing_time_ms != null ? `${result.processing_time_ms.toFixed(0)}ms` : '—'}</div>
                  <div className="metric-label">Latency</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{result.audio_duration_s?.toFixed(1)}s</div>
                  <div className="metric-label">Audio</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{result.models_used?.includes('Wav2Vec2') ? 'Wav2Vec2' : 'DSP'}</div>
                  <div className="metric-label">Model</div>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title"><IconUpload size={16} /> Upload Audio</div>
            </div>
            <CallContextForm context={context} onChange={setContext} />
            <div
              className={`upload-zone ${file ? 'has-file' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) setFile(f); }}
            >
              <IconUpload size={28} />
              <div className="upload-title">{file ? file.name : 'Drop audio file or click to browse'}</div>
              <div className="upload-formats">WAV · MP3 · OGG · FLAC · M4A (max 10 MB)</div>
            </div>
            <input ref={fileInputRef} type="file" accept="audio/*" style={{ display: 'none' }}
              onChange={e => setFile(e.target.files[0])} />
            {file && (
              <button className="btn btn-primary w-full" style={{ marginTop: 12 }}
                onClick={runAnalysis} disabled={isAnalyzing || liveMicActive}>
                {isAnalyzing ? 'Analyzing…' : 'Run Analysis'}
              </button>
            )}
            {error && <div className="notification-bar critical">{error}</div>}
          </div>

          <div className="card">
            <LiveMicMonitor
              disabled={isAnalyzing}
              callerId={context.callerId || null}
              callerName={context.callerName || null}
              apiKey={getApiKey()}
              onRiskUpdate={handleLiveRiskUpdate}
              onSessionStart={() => setLiveMicActive(true)}
              onStop={() => setLiveMicActive(false)}
              onError={msg => { setError(msg); setLiveMicActive(false); }}
            />
          </div>
        </div>

        <div className="analysis-right">
          {result && (
            <div className="notification-bar" style={{
              background: `${colors.primary}10`,
              borderColor: `${colors.primary}40`,
              color: colors.primary,
            }}>
              <SeverityDot level={result.risk_level} size={10} />
              <div>
                <div className="notif-title">{result.alert_message?.split('.')[0]}</div>
                <div className="notif-sub">{result.explanation}</div>
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <div className="card-title">Analysis Layers</div>
              {result && <AttackTypeBadge attackType={result.attack_type} />}
            </div>
            <ComponentBreakdown scores={result?.component_scores} />
          </div>

          {result && (
            <div className="card">
              <div className="card-header">
                <div className="card-title">Policy Decision</div>
                <span className="tag" style={{ background: `${colors.primary}15`, color: colors.primary }}>
                  {result.action_recommended?.replace(/_/g, ' ')}
                </span>
              </div>
              <ActionCard decision={result.action_recommended}
                verifications={getVerifications(result.action_recommended)} />
              {result.transaction_blocked && (
                <div className="txn-blocked-banner">Transaction blocked — awaiting secondary verification</div>
              )}
              {result.fraud_registry_match && (
                <div className="fraud-match-banner">{result.fraud_registry_detail}</div>
              )}
              {result.cross_session && (
                <div className="cross-session-info">{result.cross_session.explanation}</div>
              )}
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <div className="card-title">Security Alerts</div>
              {alerts.length > 0 && <span className="tag tag-critical">{alerts.length} Active</span>}
            </div>
            <AlertPanel alerts={alerts} onAcknowledge={i => setAlerts(prev => prev.filter((_, idx) => idx !== i))} />
          </div>
        </div>
      </div>
    </div>
  );
}

function SystemDiagnosticsPage() {
  const [report, setReport] = useState(null);
  const [perf, setPerf] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [benchRes, perfRes] = await Promise.all([
        apiFetch('/benchmark'),
        apiFetch('/performance'),
      ]);
      if (!benchRes.ok) throw new Error('Diagnostics unavailable');
      setReport(await benchRes.json());
      if (perfRes.ok) setPerf(await perfRes.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const m = report?.metrics;

  return (
    <div>
      <div className="stats-grid stats-grid-4">
        <StatCard value={m ? m.f1_fake_detection : '—'} label="F1 Score" sublabel="Fake detection" color="#4a9eff" />
        <StatCard value={m ? `${m.deepfake_detection_accuracy_pct}%` : '—'} label="Deepfake Accuracy" color="#2dd87b" />
        <StatCard value={m ? `${m.avg_latency_ms}ms` : '—'} label="Avg Latency" color="#ff6b3d" />
        <StatCard value={perf?.cuda_available ? 'GPU' : 'CPU'} label="Compute" sublabel={perf?.deepfake_runtime?.device || '—'} color="#f5a623" />
      </div>

      {error && <div className="notification-bar critical">{error}</div>}
      {loading && <div className="loading-center"><div className="spinner" /></div>}

      {m && (
        <div className="diag-grid">
          <div className="card">
            <div className="card-header">
              <div className="card-title">Detection Metrics</div>
              <button className="btn btn-secondary btn-sm" onClick={load} disabled={loading}>Refresh</button>
            </div>
            <div className="metrics-grid">
              {[
                ['Precision', m.precision_fake_detection],
                ['Recall', m.recall_fake_detection],
                ['F1', m.f1_fake_detection],
                ['Risk Band Acc.', `${m.risk_band_accuracy_pct}%`],
                ['True Positives', m.true_positives],
                ['True Negatives', m.true_negatives],
                ['False Positives', m.false_positives],
                ['False Negatives', m.false_negatives],
              ].map(([label, val]) => (
                <div key={label} className="metric-cell">
                  <div className="metric-cell-label">{label}</div>
                  <div className="metric-cell-value">{val}</div>
                </div>
              ))}
            </div>
            {m.language_accuracy && Object.keys(m.language_accuracy).length > 0 && (
              <div className="lang-metrics">
                <div className="lang-metrics-title">Language Coverage</div>
                <div className="lang-tags">
                  {Object.entries(m.language_accuracy).map(([lang, stats]) => (
                    <span key={lang} className="lang-tag">{lang}: {stats.accuracy_pct}%</span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header"><div className="card-title">Model Status</div></div>
            <div className="model-status-list">
              <div className="model-status-item">
                <span>Wav2Vec2 Deepfake Detector</span>
                <span className={report?.models_loaded?.wav2vec2_deepfake ? 'status-ok' : 'status-fail'}>
                  {report?.models_loaded?.wav2vec2_deepfake ? 'Loaded' : 'Fallback'}
                </span>
              </div>
              <div className="model-status-item">
                <span>ECAPA-TDNN Speaker Verifier</span>
                <span className={report?.models_loaded?.ecapa_speaker ? 'status-ok' : 'status-fail'}>
                  {report?.models_loaded?.ecapa_speaker ? 'Loaded' : 'Fallback'}
                </span>
              </div>
              <div className="model-status-item">
                <span>ONNX Runtime</span>
                <span className={perf?.onnx_available ? 'status-ok' : 'status-warn'}>
                  {perf?.onnx_available ? 'Available' : 'Not exported'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PolicyPage() {
  const [config, setConfig] = useState(null);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch('/policy')
      .then(r => r.json())
      .then(data => setConfig({
        lowThreshold: data.low_threshold ?? 30,
        highThreshold: data.high_threshold ?? 70,
        criticalThreshold: data.critical_threshold ?? 85,
        autoBlock: data.auto_block_on_high ?? true,
        requireMFA: data.require_mfa_on_medium ?? true,
        highValueLimit: data.high_value_transaction_limit_inr ?? data.high_value_transaction_limit ?? 50000,
        alertEmail: (data.alert_channels || []).includes('email'),
        alertSMS: (data.alert_channels || []).includes('sms'),
      }))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setError(null);
    const channels = ['ui'];
    if (config.alertEmail) channels.push('email');
    if (config.alertSMS) channels.push('sms');
    try {
      const res = await apiFetch('/policy', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          low_threshold: config.lowThreshold,
          high_threshold: config.highThreshold,
          critical_threshold: config.criticalThreshold,
          auto_block_on_high: config.autoBlock,
          require_mfa_on_medium: config.requireMFA,
          high_value_transaction_limit: config.highValueLimit,
          alert_channels: channels,
        }),
      });
      if (!res.ok) throw new Error('Failed to save policy');
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setError(e.message);
    }
  };

  if (loading) return <div className="loading-center"><div className="spinner" /></div>;
  if (!config) return <div className="notification-bar critical">{error || 'Failed to load policy'}</div>;

  return (
    <div className="policy-grid">
      <div className="card">
        <div className="card-header"><div className="card-title">Risk Thresholds</div></div>
        <p className="card-desc">Configure risk score boundaries. Changes apply immediately without model retraining.</p>
        {[
          { key: 'lowThreshold', label: 'LOW → MEDIUM', color: '#f5a623' },
          { key: 'highThreshold', label: 'MEDIUM → HIGH', color: '#ff6b3d' },
          { key: 'criticalThreshold', label: 'HIGH → CRITICAL', color: '#ff3355' },
        ].map(({ key, label, color }) => (
          <div key={key} className="form-group">
            <label className="form-label">{label}</label>
            <div className="slider-row">
              <input type="range" min={0} max={100} value={config[key]}
                onChange={e => setConfig(p => ({ ...p, [key]: Number(e.target.value) }))}
                style={{ flex: 1, accentColor: color }} />
              <span className="slider-value" style={{ color }}>{config[key]}</span>
            </div>
          </div>
        ))}
        <div className="form-group">
          <label className="form-label">High-Value Transaction Limit (INR)</label>
          <input className="form-input" type="number" value={config.highValueLimit}
            onChange={e => setConfig(p => ({ ...p, highValueLimit: Number(e.target.value) }))} />
        </div>
        {error && <div className="form-error">{error}</div>}
        <button className="btn btn-primary w-full" onClick={handleSave}>
          {saved ? 'Policy Saved' : 'Save Policy'}
        </button>
      </div>

      <div className="policy-right">
        <div className="card">
          <div className="card-header"><div className="card-title">Alert Channels</div></div>
          {[
            { key: 'autoBlock', label: 'Auto-block on HIGH risk' },
            { key: 'requireMFA', label: 'Require MFA on MEDIUM risk' },
            { key: 'alertEmail', label: 'Email alerts' },
            { key: 'alertSMS', label: 'SMS alerts' },
          ].map(({ key, label }) => (
            <div key={key} className="toggle-row">
              <span>{label}</span>
              <div className={`toggle ${config[key] ? 'on' : ''}`}
                onClick={() => setConfig(p => ({ ...p, [key]: !p[key] }))} />
            </div>
          ))}
        </div>
        <div className="card">
          <div className="card-header"><div className="card-title">Policy Preview</div></div>
          {[
            { range: `0 – ${config.lowThreshold}`, level: 'LOW', action: 'Allow', color: '#2dd87b' },
            { range: `${config.lowThreshold} – ${config.highThreshold}`, level: 'MEDIUM', action: 'Step-Up MFA', color: '#f5a623' },
            { range: `${config.highThreshold} – ${config.criticalThreshold}`, level: 'HIGH', action: 'Block + Alert', color: '#ff6b3d' },
            { range: `${config.criticalThreshold} – 100`, level: 'CRITICAL', action: 'Block + Escalate', color: '#ff3355' },
          ].map(row => (
            <div key={row.level} className="policy-row" style={{ '--row-color': row.color }}>
              <span className="mono">{row.range}</span>
              <span className="policy-level">{row.level}</span>
              <span>{row.action}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function buildAlerts(result) {
  const alerts = [];
  const time = new Date().toLocaleString('en-IN');

  if (result.risk_level === 'CRITICAL' || result.risk_level === 'HIGH') {
    alerts.push({
      level: result.risk_level,
      title: result.risk_level === 'CRITICAL' ? 'Voice Cloning Attack Detected' : 'Voice Manipulation Suspected',
      message: result.alert_message,
      actions: getVerifications(result.action_recommended),
      channels: result.channel_notifications || [],
      time,
    });
  }
  if (result.fraud_registry_match) {
    alerts.push({
      level: 'HIGH',
      title: 'Fraud Registry Match',
      message: result.fraud_registry_detail || 'Caller found in fraud watchlist.',
      actions: ['Block transaction', 'Notify compliance', 'Log incident'],
      channels: [],
      time,
    });
  }
  if (result.cross_session?.drift_detected) {
    alerts.push({
      level: 'MEDIUM',
      title: 'Cross-Session Speaker Drift',
      message: result.cross_session.explanation,
      actions: ['Require callback verification'],
      channels: [],
      time,
    });
  }
  if (result.transaction_blocked) {
    alerts.push({
      level: 'CRITICAL',
      title: 'Transaction Blocked',
      message: `Transaction blocked. Risk score ${result.risk_score}/100 exceeded policy threshold.`,
      actions: ['End call', 'Verify via registered callback', 'Notify compliance'],
      channels: (result.channel_notifications || []).filter(c => c.channel === 'sms'),
      time,
    });
  }
  (result.channel_notifications || []).forEach(n => {
    if (n.channel !== 'ui') {
      alerts.push({
        level: result.risk_level,
        title: `${n.channel.toUpperCase()} Notification`,
        message: `${n.recipient}: ${n.message}`,
        actions: [],
        channels: [n],
        time,
      });
    }
  });
  return alerts;
}

function getVerifications(action) {
  const map = {
    ESCALATED: ['Notify supervisor', 'Callback via registered device', 'Dual-factor authentication', 'Log incident'],
    BLOCKED: ['End call and callback', 'OTP verification', 'Log incident'],
    STEP_UP_VERIFICATION: ['OTP verification on registered mobile'],
    ALLOWED: ['Proceed with standard verification'],
  };
  return map[action] || [];
}

function Sidebar({ activePage, setActivePage, onLogout }) {
  const nav = [
    { id: 'live', label: 'Voice Analysis', icon: IconMic },
    { id: 'incidents', label: 'Incident Log', icon: IconList },
    { id: 'policy', label: 'Policy Config', icon: IconSettings },
    { id: 'system', label: 'System Diagnostics', icon: IconActivity },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-mark">
          <IconShield size={22} />
          <span className="logo-text">VoiceShield</span>
        </div>
        <div className="logo-sub">Voice Integrity Platform</div>
        <div className="logo-org">Cyber Security Cell</div>
      </div>
      <nav className="sidebar-nav">
        {nav.map(item => (
          <button key={item.id}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => setActivePage(item.id)}>
            <item.icon size={16} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="system-status">
          <div className="status-dot" />
          <div>
            <div className="status-label">System Online</div>
            <div className="status-sub">6 detection layers active</div>
          </div>
        </div>
        <button className="btn btn-ghost btn-sm logout-btn" onClick={onLogout}>
          <IconLogout size={14} /> Sign Out
        </button>
      </div>
    </aside>
  );
}

export default function App() {
  const [authenticated, setAuthenticated] = useState(!!getApiKey());
  const [authRequired, setAuthRequired] = useState(false);
  const [activePage, setActivePage] = useState('live');
  const [incidents, setIncidents] = useState([]);
  const [isLoadingIncidents, setIsLoadingIncidents] = useState(false);
  const [stats, setStats] = useState({ total: 0, highRisk: 0, blocked: 0, avgScore: '—' });
  const [backendOnline, setBackendOnline] = useState(true);

  const fetchIncidents = useCallback(async () => {
    if (!getApiKey() && authRequired) return;
    setIsLoadingIncidents(true);
    try {
      const [incRes, statsRes] = await Promise.all([
        apiFetch('/incidents'),
        apiFetch('/stats'),
      ]);
      if (incRes.ok) {
        const data = await incRes.json();
        setIncidents(data.incidents || []);
      }
      if (statsRes.ok) {
        const s = await statsRes.json();
        setStats({
          total: s.total_calls_analyzed ?? 0,
          highRisk: s.high_risk_calls ?? 0,
          blocked: s.transactions_blocked ?? 0,
          avgScore: s.average_risk_score ?? '—',
        });
      }
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    } finally {
      setIsLoadingIncidents(false);
    }
  }, [authRequired]);

  const addIncident = useCallback((inc) => {
    setIncidents(prev => [inc, ...prev]);
    setStats(prev => ({
      total: prev.total + 1,
      highRisk: prev.highRisk + (inc.risk_score >= 70 ? 1 : 0),
      blocked: prev.blocked + (inc.transaction_blocked ? 1 : 0),
      avgScore: '—',
    }));
  }, []);

  useEffect(() => {
    checkHealth().then(ok => {
      if (!ok) { setBackendOnline(false); return; }
      apiFetch('/stats')
        .then(r => { if (r.status === 401) setAuthRequired(true); else { setAuthRequired(false); if (getApiKey()) fetchIncidents(); } })
        .catch(() => setBackendOnline(false));
    });
  }, [fetchIncidents]);

  useEffect(() => {
    if (authenticated) fetchIncidents();
  }, [authenticated, fetchIncidents]);

  const handleLogout = () => {
    clearApiKey();
    setAuthenticated(false);
  };

  if (authRequired && !authenticated) {
    return <LoginPage onLogin={() => { setAuthenticated(true); fetchIncidents(); }} />;
  }

  return (
    <div className="app-layout">
      <Sidebar activePage={activePage} setActivePage={setActivePage} onLogout={handleLogout} />
      <div className="main-content">
        <header className="topbar">
          <div>
            <h1 className="topbar-title">{PAGE_TITLES[activePage]}</h1>
            <p className="topbar-subtitle">Cyber Security Cell — Voice Cloning Defense</p>
          </div>
          <div className="topbar-right">
            {!backendOnline && <span className="topbar-chip chip-warn">Backend Offline</span>}
            <span className="topbar-chip">Privacy-First</span>
            <div className="live-indicator">
              <div className="live-dot" />
              Active Monitoring
            </div>
          </div>
        </header>
        <main className="page-content">
          {activePage === 'live' && <LiveAnalysisPage stats={stats} onNewIncident={addIncident} />}
          {activePage === 'incidents' && (
            <div className="card">
              <div className="card-header">
                <div className="card-title">Security Incidents</div>
                <span className="tag tag-info">{incidents.length} records</span>
              </div>
              <IncidentLog incidents={incidents} isLoading={isLoadingIncidents} />
            </div>
          )}
          {activePage === 'policy' && <PolicyPage />}
          {activePage === 'system' && <SystemDiagnosticsPage />}
        </main>
      </div>
    </div>
  );
}
