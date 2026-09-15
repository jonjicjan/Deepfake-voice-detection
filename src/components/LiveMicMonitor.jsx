// VoiceShield AI — Real-time microphone streaming via WebSocket
import { useState, useRef, useCallback, useEffect } from 'react';
import { IconMic } from '../icons';

const SAMPLE_RATE = 16000;

function buildWsUrl(apiKey) {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const base = `${proto}://${window.location.host}/api/ws/stream`;
  return apiKey ? `${base}?api_key=${encodeURIComponent(apiKey)}` : base;
}

export function LiveMicMonitor({
  onRiskUpdate,
  onSessionStart,
  onStop,
  onError,
  callerId = null,
  callerName = null,
  apiKey = '',
  disabled = false,
}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [chunkCount, setChunkCount] = useState(0);
  const [status, setStatus] = useState('idle');

  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const isStreamingRef = useRef(false);

  const cleanup = useCallback(() => {
    isStreamingRef.current = false;
    processorRef.current?.disconnect();
    processorRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.close();
    wsRef.current = null;
    setIsStreaming(false);
    setStatus('idle');
  }, []);

  useEffect(() => () => cleanup(), [cleanup]);

  const startStreaming = useCallback(async () => {
    if (isStreamingRef.current || disabled) return;
    try {
      setStatus('connecting');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      streamRef.current = stream;

      const ws = new WebSocket(buildWsUrl(apiKey));
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'stream_config',
          caller_id: callerId,
          caller_name: callerName,
          call_origin: 'LIVE_STREAM',
        }));
        onSessionStart?.();
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'session_start') setSessionId(data.session_id);
        if (data.type === 'risk_update') {
          setChunkCount(data.chunk);
          onRiskUpdate?.(data);
        }
        if (data.type === 'error' || data.type === 'analysis_error') onError?.(data.message);
      };

      ws.onerror = () => onError?.('WebSocket connection failed');
      ws.onclose = () => { if (isStreamingRef.current) cleanup(); };

      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!isStreamingRef.current || ws.readyState !== WebSocket.OPEN) return;
        const float32 = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 32768 : s * 32767;
        }
        ws.send(int16.buffer);
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      isStreamingRef.current = true;
      setIsStreaming(true);
      setStatus('monitoring');
    } catch (e) {
      onError?.(e.message || 'Microphone access denied');
      cleanup();
    }
  }, [apiKey, callerId, callerName, cleanup, disabled, onError, onRiskUpdate, onSessionStart]);

  const stopStreaming = useCallback(() => {
    cleanup();
    setChunkCount(0);
    setSessionId(null);
    onStop?.();
  }, [cleanup, onStop]);

  return (
    <div className="live-mic-panel">
      <div className="card-header" style={{ marginBottom: 12 }}>
        <div className="card-title"><IconMic size={16} /> Live Microphone Monitor</div>
        {isStreaming && <span className="tag tag-critical live-tag">LIVE</span>}
      </div>
      <p className="live-mic-desc">
        Streams audio for real-time deepfake and speaker analysis. Risk score updates every 3 seconds.
      </p>
      {!isStreaming ? (
        <button className="btn btn-primary w-full" onClick={startStreaming} disabled={disabled}>
          Start Live Monitor
        </button>
      ) : (
        <button className="btn btn-secondary w-full" onClick={stopStreaming}>Stop Monitoring</button>
      )}
      {isStreaming && (
        <div className="live-status-box">
          <div>Status: <strong>{status.toUpperCase()}</strong></div>
          {sessionId && <div className="mono subtle">Session: {sessionId.slice(0, 8)}</div>}
          {chunkCount > 0 && <div>Chunks analyzed: {chunkCount}</div>}
        </div>
      )}
    </div>
  );
}
