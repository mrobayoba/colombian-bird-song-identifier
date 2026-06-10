import { useEffect, useRef } from "react";

interface WaveformProps {
  historyRef: React.MutableRefObject<number[]>;
  active: boolean;
}

/**
 * The signature element: a live, mirrored waveform that breathes with the
 * bird's song as it is recorded. Bars rise from a central axis; the leading
 * edge carries the tanager accent so the eye follows the newest sound.
 */
export function Waveform({ historyRef, active }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    };
    resize();
    window.addEventListener("resize", resize);

    const styles = getComputedStyle(document.documentElement);
    const tanager = styles.getPropertyValue("--tanager").trim() || "#f0531c";
    const moss = styles.getPropertyValue("--moss-400").trim() || "#5c9a86";
    const dim = styles.getPropertyValue("--forest-600").trim() || "#1b6055";

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const hist = historyRef.current;
      const n = hist.length;
      const gap = 2 * dpr;
      const barW = (w - gap * (n - 1)) / n;
      const mid = h / 2;

      for (let i = 0; i < n; i++) {
        const v = hist[i];
        const barH = Math.max(2 * dpr, v * h * 0.92);
        const x = i * (barW + gap);
        const isLeading = i >= n - 3;
        if (!active && v === 0) {
          ctx.fillStyle = dim;
        } else if (isLeading && active) {
          ctx.fillStyle = tanager;
        } else {
          // fade older bars toward moss
          const age = i / n;
          ctx.fillStyle = age > 0.5 ? moss : dim;
        }
        // rounded mirrored bar
        const r = Math.min(barW / 2, 3 * dpr);
        roundRect(ctx, x, mid - barH / 2, barW, barH, r);
        ctx.fill();
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [historyRef, active]);

  return <canvas ref={canvasRef} className="waveform" aria-hidden="true" />;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
