import type { UnifiedResponse } from "../lib/types";

interface ResultCardProps {
  result: UnifiedResponse;
  onReset: () => void;
}

function confidencePct(c: number): string {
  return `${Math.round(c * 100)}%`;
}

function statusColor(code: string): string {
  if (["CR", "EN"].includes(code)) return "var(--cr)";
  if (["VU", "NT"].includes(code)) return "var(--vu)";
  if (code === "LC") return "var(--lc)";
  return "var(--parchment-dim)";
}

export function ResultCard({ result, onReset }: ResultCardProps) {
  const { identification, card, enrichment, degraded } = result;
  const top = identification;
  const conf = Math.round(top.confidence * 100);
  const lowConfidence = top.confidence < 0.4;

  return (
    <div className="result">
      {card?.image_url ? (
        <div className="result-hero">
          <img src={card.image_url} alt={top.display_name} loading="lazy" />
          <div className="result-hero-fade" />
        </div>
      ) : null}

      <div className="result-headrow">
        <span className="result-eyebrow">
          {lowConfidence ? "Coincidencia posible" : "Identificación"}
        </span>
        <span className="result-conf" data-low={lowConfidence}>
          {confidencePct(top.confidence)}
        </span>
      </div>

      <h1 className="result-name">{top.display_name}</h1>
      {card?.english_name ? (
        <p className="result-common">
          {card.english_name}
          {card.taxonomy?.family ? ` · ${card.taxonomy.family}` : ""}
        </p>
      ) : null}

      <div className="conf-bar" aria-label={`Confianza ${conf}%`}>
        <span style={{ width: `${conf}%` }} />
      </div>

      {card?.conservation_status &&
      card.conservation_status.code !== "NE" ? (
        <div
          className="status-chip"
          style={{ ["--chip" as string]: statusColor(card.conservation_status.code) }}
        >
          <span className="status-dot" />
          {card.conservation_status.label_es ||
            card.conservation_status.code}
        </div>
      ) : null}

      {enrichment?.summary_es ? (
        <p className="result-summary">{enrichment.summary_es}</p>
      ) : card?.description ? (
        <p className="result-summary">{card.description}</p>
      ) : null}

      {enrichment?.semantic_tags?.length ? (
        <div className="tags">
          {enrichment.semantic_tags.map((t) => (
            <span className="tag" key={t}>
              {t}
            </span>
          ))}
        </div>
      ) : null}

      {top.alternatives.length > 0 ? (
        <div className="alts">
          <span className="alts-label">También podría ser</span>
          {top.alternatives.map((a) => (
            <div className="alt-row" key={a.label}>
              <span className="alt-name">{a.display_name}</span>
              <span className="alt-conf">{confidencePct(a.confidence)}</span>
            </div>
          ))}
        </div>
      ) : null}

      <div className="result-meta">
        <span>{top.n_chunks} fragmentos analizados</span>
        {degraded.includes("enrichment") ? (
          <span className="meta-warn">descripción no disponible</span>
        ) : null}
        {degraded.includes("card") ? (
          <span className="meta-warn">ficha no disponible</span>
        ) : null}
      </div>

      <button className="btn-again" onClick={onReset}>
        Grabar otra ave
      </button>
    </div>
  );
}
