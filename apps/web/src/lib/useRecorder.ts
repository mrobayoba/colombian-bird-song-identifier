import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderState = "idle" | "recording" | "stopped" | "denied";

interface UseRecorderResult {
  state: RecorderState;
  seconds: number;
  /** Live normalized amplitude 0..1, updated ~60fps while recording. */
  amplitudeRef: React.MutableRefObject<number>;
  /** Rolling history of amplitudes for drawing a waveform. */
  historyRef: React.MutableRefObject<number[]>;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
  reset: () => void;
}

const HISTORY_LEN = 96; // number of bars in the waveform

export function useRecorder(): UseRecorderResult {
  const [state, setState] = useState<RecorderState>("idle");
  const [seconds, setSeconds] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);

  const amplitudeRef = useRef(0);
  const historyRef = useRef<number[]>(new Array(HISTORY_LEN).fill(0));

  const cleanup = useCallback(() => {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    if (timerRef.current != null) clearInterval(timerRef.current);
    rafRef.current = null;
    timerRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    analyserRef.current = null;
    streamRef.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const tick = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const buf = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(buf);
    // RMS around the 128 midpoint → 0..1
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / buf.length);
    const level = Math.min(1, rms * 2.2); // gentle boost for visibility
    amplitudeRef.current = level;
    const h = historyRef.current;
    h.push(level);
    if (h.length > HISTORY_LEN) h.shift();
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      historyRef.current = new Array(HISTORY_LEN).fill(0);

      // Visualization graph
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const ctx = new AudioCtx();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      audioCtxRef.current = ctx;
      analyserRef.current = analyser;

      // Recording
      const mr = new MediaRecorder(stream);
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorderRef.current = mr;
      mr.start();

      setSeconds(0);
      setState("recording");
      timerRef.current = window.setInterval(
        () => setSeconds((s) => s + 1),
        1000,
      );
      rafRef.current = requestAnimationFrame(tick);
    } catch {
      setState("denied");
    }
  }, [tick]);

  const stop = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const mr = mediaRecorderRef.current;
      if (!mr || mr.state === "inactive") {
        resolve(null);
        return;
      }
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: mr.mimeType || "audio/webm",
        });
        cleanup();
        setState("stopped");
        resolve(blob);
      };
      mr.stop();
    });
  }, [cleanup]);

  const reset = useCallback(() => {
    cleanup();
    chunksRef.current = [];
    amplitudeRef.current = 0;
    historyRef.current = new Array(HISTORY_LEN).fill(0);
    setSeconds(0);
    setState("idle");
  }, [cleanup]);

  return { state, seconds, amplitudeRef, historyRef, start, stop, reset };
}
