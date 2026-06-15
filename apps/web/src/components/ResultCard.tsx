import type { UnifiedResponse } from "../lib/types";

interface ResultCardProps {
  result: UnifiedResponse;
  onReset: () => void;
}

function confidencePct(c: number): string {
  return `${(c * 100).toFixed(2)}%`;
}

function statusColor(code: string): string {
  if (["CR", "EN"].includes(code)) return "var(--cr)";
  if (["VU", "NT"].includes(code)) return "var(--vu)";
  if (code === "LC") return "var(--lc)";
  return "var(--parchment-dim)";
}

function readMoreHref(
  infoUrl: string | undefined,
  target: string | undefined
): string | null {
  if (infoUrl) return infoUrl;
  if (target)
    return `https://es.wikipedia.org/w/index.php?search=${encodeURIComponent(target)}`;
  return null;
}

export function ResultCard({ result, onReset }: ResultCardProps) {
  const { identification, card, enrichment, degraded } = result;
  const top = identification;
  const conf = top.confidence * 100;
  const lowConfidence = top.confidence < 0.4;

  const href = readMoreHref(card?.info_url, enrichment?.read_more_target);

  const taxonomyLine = card
    ? [card.english_name, card.taxonomy?.family, card.taxonomy?.order]
        .filter(Boolean)
        .join(" · ")
    : null;

  const hasStats =
    card && (card.inat_sightings > 0 || card.xeno_canto_recordings > 0);

  const hasConservation =
    card?.conservation_status && card.conservation_status.code !== "NE";

  return (
    <div className="result">
      {/* ── Columna izquierda: imagen fija ── */}
      <div className="result-hero">
        {card?.image_url ? (
          <img src={card.image_url} alt={top.display_name} loading="lazy" />
        ) : (
          <div className="result-hero-placeholder" />
        )}
        <div className="result-hero-fade" />
      </div>

      {/* ── Columna derecha: toda la información ── */}
      <div className="result-content">
        <div className="result-headrow">
          <span className="result-eyebrow">
            {lowConfidence ? "Coincidencia posible" : "Identificación"}
          </span>
          <span className="result-conf" data-low={lowConfidence}>
            {confidencePct(top.confidence)}
          </span>
        </div>

        <h1 className="result-name">{top.display_name}</h1>
        {taxonomyLine ? (
          <p className="result-common">{taxonomyLine}</p>
        ) : null}

        <div className="conf-bar" aria-label={`Confianza ${confidencePct(top.confidence)}`}>
          <span style={{ width: `${conf}%` }} />
        </div>

        {hasConservation || hasStats ? (
          <div className="result-status-row">
            {hasConservation ? (
              <div
                className="status-chip"
                style={{
                  ["--chip" as string]: statusColor(
                    card!.conservation_status.code
                  ),
                }}
              >
                <span className="status-dot" />
                {card!.conservation_status.label_es ||
                  card!.conservation_status.code}
              </div>
            ) : (
              <div />
            )}
            {hasStats ? (
              <div className="result-stats">
                {card!.inat_sightings > 0 ? (
                  <div className="result-stat">
                    <span className="stat-value">
                      {card!.inat_sightings.toLocaleString("es-CO")}
                    </span>
                    <span className="stat-label">iNaturalist</span>
                  </div>
                ) : null}
                {card!.xeno_canto_recordings > 0 ? (
                  <div className="result-stat">
                    <span className="stat-value">
                      {card!.xeno_canto_recordings.toLocaleString("es-CO")}
                    </span>
                    <span className="stat-label">Xeno-canto</span>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}

        {enrichment?.summary_es ? (
          <p className="result-summary">{enrichment.summary_es}</p>
        ) : card?.description ? (
          <p className="result-summary">{card.description}</p>
        ) : null}

        {enrichment?.behavior_notes ? (
          <div className="behavior-card">
            <span className="behavior-label">Comportamiento</span>
            <p className="behavior-text">{enrichment.behavior_notes}</p>
          </div>
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

        {enrichment?.alternatives?.length ? (
          <div className="related-species">
            <span className="related-label">Especies relacionadas</span>
            {enrichment.alternatives.map((name) => (
              <div className="related-row" key={name}>
                <span className="related-dot" />
                <span className="related-name">{name}</span>
              </div>
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

        <div
          className={`result-actions${!href ? " result-actions--solo" : ""}`}
        >
          {href ? (
            <a
              className="btn-read-more"
              href={href}
              target="_blank"
              rel="noopener noreferrer"
            >
              Leer más
            </a>
          ) : null}
          <button className="btn-again" onClick={onReset}>
            Grabar otra ave
          </button>
        </div>

        <div className="result-meta">
          <span>{top.n_chunks} fragmentos analizados</span>
          {degraded.includes("enrichment") ? (
            <span className="meta-warn">descripción no disponible</span>
          ) : null}
          {degraded.includes("card") ? (
            <span className="meta-warn">ficha no disponible</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}
