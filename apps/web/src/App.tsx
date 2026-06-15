import { useEffect, useState } from "react";
import { Waveform } from "./components/Waveform";
import { ResultCard } from "./components/ResultCard";
import { useRecorder } from "./lib/useRecorder";
import { identify, ApiError } from "./lib/api";
import type { UnifiedResponse } from "./lib/types";
import "./App.css";

type Phase = "ready" | "recording" | "loading" | "result" | "error";

const MIN_SECONDS = 3; // model needs >3s to extract a chunk

export function App() {
  const rec = useRecorder();
  const [phase, setPhase] = useState<Phase>("ready");
  const [result, setResult] = useState<UnifiedResponse | null>(null);
  const [error, setError] = useState<string>("");

  // Keep phase in sync with a denied-permission recorder state.
  useEffect(() => {
    if (rec.state === "denied") {
      setError(
        "No se pudo acceder al micrófono. Revisa los permisos del navegador.",
      );
      setPhase("error");
    }
  }, [rec.state]);

  const handleStart = async () => {
    setError("");
    await rec.start();
    setPhase("recording");
  };

  const handleStop = async () => {
    const blob = await rec.stop();
    if (!blob) {
      setError("No se capturó audio. Intenta de nuevo.");
      setPhase("error");
      return;
    }
    setPhase("loading");
    try {
      const res = await identify(blob);
      setResult(res);
      setPhase("result");
    } catch (e) {
      const msg =
        e instanceof ApiError ? e.message : "Algo salió mal. Intenta de nuevo.";
      setError(msg);
      setPhase("error");
    }
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // reset so the same file can be chosen again
    if (!file) return;
    setError("");
    setPhase("loading");
    try {
      const res = await identify(file);
      setResult(res);
      setPhase("result");
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? err.message
          : "Algo salió mal. Intenta de nuevo.";
      setError(msg);
      setPhase("error");
    }
  };

  const handleReset = () => {
    rec.reset();
    setResult(null);
    setError("");
    setPhase("ready");
  };

  const canStop = rec.seconds >= MIN_SECONDS;

  return (
    <div className="app">
      <header className="masthead">
        <span className="mark">Cantos</span>
        <span className="mark-sub">aves de Colombia</span>
      </header>

      {phase === "result" && result ? (
        <ResultCard result={result} onReset={handleReset} />
      ) : (
        <main className="stage">
          {phase === "error" ? (
            <div className="panel">
              <p className="error-text">{error}</p>
              <button className="btn-again" onClick={handleReset}>
                Volver a intentar
              </button>
            </div>
          ) : phase === "loading" ? (
            <div className="panel loading">
              <div className="pulse-rings" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <p className="loading-text">Escuchando el canto…</p>
              <p className="loading-sub">Analizando frecuencias</p>
            </div>
          ) : (
            <div className="panel">
              <Waveform
                historyRef={rec.historyRef}
                active={phase === "recording"}
              />

              {phase === "recording" ? (
                <p className="timer">
                  {String(Math.floor(rec.seconds / 60)).padStart(1, "0")}:
                  {String(rec.seconds % 60).padStart(2, "0")}
                </p>
              ) : (
                <p className="prompt">
                  Apunta el micrófono hacia el ave y graba su canto
                </p>
              )}

              {phase === "ready" ? (
                <>
                  <button className="btn-record" onClick={handleStart}>
                    <span className="btn-record-dot" />
                    Grabar
                  </button>
                  <label className="btn-upload">
                    Subir un audio
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={handleFile}
                      hidden
                    />
                  </label>
                </>
              ) : (
                <button
                  className="btn-stop"
                  onClick={handleStop}
                  disabled={!canStop}
                >
                  {canStop
                    ? "Identificar"
                    : `Sigue grabando… ${MIN_SECONDS - rec.seconds}s`}
                </button>
              )}
            </div>
          )}
        </main>
      )}

      <footer className="footnote">
        <div className="footnote-col">
          <span className="footnote-title">Cantos</span>
          <span className="footnote-item">Identificador de aves de Colombia</span>
          <span className="footnote-item">v0.1.0 · CBSI</span>
        </div>
        <div className="footnote-col">
          <span className="footnote-title">Fuentes</span>
          <span className="footnote-item">Xeno-Canto</span>
          <span className="footnote-item">iNaturalist</span>
          <span className="footnote-item">IUCN Red List</span>
        </div>
        <div className="footnote-col">
          <span className="footnote-title">Acerca</span>
          <span className="footnote-item">Proyecto CBSI</span>
          <span className="footnote-item">Código abierto</span>
        </div>
      </footer>
    </div>
  );
}
