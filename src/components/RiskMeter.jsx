// VoiceShield AI — Risk Meter
// Animated SVG arc gauge with risk color coding

import { useEffect, useRef } from 'react';

const COLORS = {
  LOW:      { primary: '#2dd87b', glow: 'rgba(45,216,123,0.3)', text: '#2dd87b', label: '✅ SAFE' },
  MEDIUM:   { primary: '#f5a623', glow: 'rgba(245,166,35,0.3)', text: '#f5a623', label: '🟡 CAUTION' },
  HIGH:     { primary: '#ff6b3d', glow: 'rgba(255,107,61,0.3)', text: '#ff6b3d', label: '🔴 HIGH RISK' },
  CRITICAL: { primary: '#ff3355', glow: 'rgba(255,51,85,0.35)', text: '#ff3355', label: '🚨 CRITICAL' },
};

export function RiskMeter({ score = 0, level = 'LOW', size = 220 }) {
  const prevScoreRef = useRef(0);
  const animScoreRef = useRef(0);
  const rafRef = useRef(null);
  const svgRef = useRef(null);

  const colors = COLORS[level] || COLORS.LOW;

  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.38;
  const strokeW = size * 0.07;

  // Arc: 210° sweep, starting from 195° (bottom-left)
  const startAngleDeg = 195;
  const sweepDeg = 210;

  const polarToXY = (angleDeg, radius) => {
    const rad = ((angleDeg - 90) * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  };

  const arcPath = (fromDeg, toDeg, rad) => {
    const start = polarToXY(fromDeg, rad);
    const end = polarToXY(toDeg, rad);
    const large = toDeg - fromDeg > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${rad} ${rad} 0 ${large} 1 ${end.x} ${end.y}`;
  };

  const endAngle = startAngleDeg + (score / 100) * sweepDeg;
  const needleAngle = startAngleDeg + (score / 100) * sweepDeg;
  const needle = polarToXY(needleAngle, r * 0.82);

  return (
    <div className="risk-gauge-wrapper" style={{ '--gauge-glow': colors.glow }}>
      <svg
        ref={svgRef}
        className="risk-gauge-svg"
        width={size}
        height={size * 0.82}
        viewBox={`0 0 ${size} ${size * 0.82}`}
        aria-label={`Risk score: ${score} out of 100`}
      >
        <defs>
          <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#2dd87b" />
            <stop offset="40%" stopColor="#f5a623" />
            <stop offset="70%" stopColor="#ff6b3d" />
            <stop offset="100%" stopColor="#ff3355" />
          </linearGradient>
          <filter id="gaugeGlow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Track */}
        <path
          d={arcPath(startAngleDeg, startAngleDeg + sweepDeg, r)}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />

        {/* Tick marks */}
        {[0, 25, 50, 75, 100].map(pct => {
          const angle = startAngleDeg + (pct / 100) * sweepDeg;
          const inner = polarToXY(angle, r - strokeW * 0.8);
          const outer = polarToXY(angle, r + strokeW * 0.8);
          return (
            <line
              key={pct}
              x1={inner.x} y1={inner.y}
              x2={outer.x} y2={outer.y}
              stroke="rgba(255,255,255,0.15)"
              strokeWidth={1.5}
              strokeLinecap="round"
            />
          );
        })}

        {/* Active arc */}
        {score > 0 && (
          <path
            d={arcPath(startAngleDeg, endAngle, r)}
            fill="none"
            stroke="url(#arcGrad)"
            strokeWidth={strokeW}
            strokeLinecap="round"
            filter="url(#gaugeGlow)"
            style={{ transition: 'all 0.6s cubic-bezier(0.4,0,0.2,1)' }}
          />
        )}

        {/* Needle dot */}
        {score > 0 && (
          <circle
            cx={needle.x}
            cy={needle.y}
            r={strokeW * 0.45}
            fill={colors.primary}
            filter="url(#gaugeGlow)"
            style={{ transition: 'all 0.6s cubic-bezier(0.4,0,0.2,1)' }}
          />
        )}

        {/* Center score text */}
        <text
          x={cx} y={cy * 0.92}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={colors.text}
          fontSize={size * 0.19}
          fontWeight="900"
          fontFamily="Inter, sans-serif"
          style={{ transition: 'fill 0.4s' }}
        >
          {Math.round(score)}
        </text>
        <text
          x={cx} y={cy * 1.12}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(255,255,255,0.35)"
          fontSize={size * 0.07}
          fontWeight="600"
          fontFamily="Inter, sans-serif"
          letterSpacing="1"
        >
          / 100
        </text>

        {/* Level labels at ends */}
        <text x={polarToXY(startAngleDeg, r + strokeW * 1.8).x} y={polarToXY(startAngleDeg, r + strokeW * 1.8).y}
          textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize={size * 0.055} fontFamily="Inter, sans-serif" fontWeight="600">0</text>
        <text x={polarToXY(startAngleDeg + sweepDeg, r + strokeW * 1.8).x} y={polarToXY(startAngleDeg + sweepDeg, r + strokeW * 1.8).y}
          textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize={size * 0.055} fontFamily="Inter, sans-serif" fontWeight="600">100</text>
      </svg>

      {/* Level badge */}
      <div
        className={`risk-level-badge ${level}`}
        style={{ transition: 'all 0.4s' }}
      >
        {colors.label}
      </div>
    </div>
  );
}
