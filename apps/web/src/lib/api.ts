import type { UnifiedResponse } from "./types";

// In dev, leave VITE_API_BASE unset — Vite proxies /api to localhost:8000.
// In production, set VITE_API_BASE to your API origin, e.g. https://api.cantos.co
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

/**
 * Send a recording to the Bird Index Agent and get a unified identification.
 * The backend expects a multipart field named exactly "audio".
 */
export async function identify(audio: Blob): Promise<UnifiedResponse> {
  const form = new FormData();
  // filename hints the container type; the backend re-decodes regardless.
  const ext = audio.type.includes("wav") ? "wav" : "webm";
  form.append("audio", audio, `recording.${ext}`);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/v1/identify`, {
      method: "POST",
      body: form,
    });
  } catch {
    // Network-level failure (server down, offline, DNS).
    throw new ApiError(0, "No se pudo conectar con el servidor.");
  }

  if (!res.ok) {
    let detail = `Error ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* response had no JSON body */
    }
    if (res.status === 413) detail = "La grabación es demasiado larga.";
    if (res.status === 502) detail = "No se pudo identificar el ave. Intenta una grabación más clara.";
    throw new ApiError(res.status, detail);
  }

  return (await res.json()) as UnifiedResponse;
}

/** Quick liveness check used by the footer status dot. */
export async function health(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
