// VoiceShield AI — Waveform Visualizer
// Animated audio waveform with color coding by risk level

import { useEffect, useRef, useState } from 'react';

export function WaveformVisualizer({ isActive = false, riskLevel = 'LOW', barCount = 48 }) {
  const [bars, setBars] = useState(() => Array(barCount).fill(0.1));
  const animRef = useRef(null);
  const timeRef = useRef(0);

  const colorMap = {
    LOW:      { base: '#2dd87b', glow: 'rgba(45,216,123,0.6)' },
    MEDIUM:   { base: '#f5a623', glow: 'rgba(245,166,35,0.6)' },
    HIGH:     { base: '#ff6b3d', glow: 'rgba(255,107,61,0.6)' },
    CRITICAL: { base: '#ff3355', glow: 'rgba(255,51,85,0.6)' },
  };
  const color = colorMap[riskLevel] || colorMap.LOW;

  useEffect(() => {
    if (!isActive) {
      setBars(Array(barCount).fill(0.05));
      return;
    }

    const animate = () => {
      timeRef.current += 0.04;
      const t = timeRef.current;
      const newBars = Array.from({ length: barCount }, (_, i) => {
        const phase = (i / barCount) * Math.PI * 2;
        const wave1 = Math.sin(t * 2.1 + phase) * 0.35;
        const wave2 = Math.sin(t * 3.7 + phase * 1.4) * 0.25;
        const wave3 = Math.sin(t * 5.3 + phase * 0.8) * 0.15;
        const noise = (Math.random() - 0.5) * 0.12;
        const val = Math.abs(wave1 + wave2 + wave3 + noise);
        return Math.max(0.06, Math.min(1.0, val));
      });
      setBars(newBars);
      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, [isActive, barCount]);

  return (
    <div className="waveform-container" role="img" aria-label="Audio waveform visualization">
      {bars.map((height, i) => (
        <div
          key={i}
          className="waveform-bar"
          style={{
            height: `${height * 100}%`,
            background: isActive
              ? `linear-gradient(to top, ${color.base}, ${color.glow})`
              : 'rgba(255,255,255,0.08)',
            boxShadow: isActive && height > 0.5 ? `0 0 6px ${color.glow}` : 'none',
            opacity: isActive ? 0.85 + height * 0.15 : 0.3,
          }}
        />
      ))}
    </div>
  );
}
