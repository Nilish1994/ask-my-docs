"use client";

import { useState } from "react";

import type { Hit, Mode } from "@/lib/api";
import { MODE_LABELS } from "@/lib/api";

function Bar({ label, value, max, display, rank }: {
  label: string;
  value: number;
  max: number;
  display: string;
  rank: number | null;
}) {
  const width = max > 0 ? Math.max(2, Math.round((value / max) * 100)) : 0;
  return (
    <div className="bar-row" title={rank ? `${label} rank #${rank}` : `${label}: not in its top 50`}>
      <span className="bar-label">{label}</span>
      <span className="bar">
        <span className="bar-fill" style={{ width: `${width}%` }} />
      </span>
      <span className="bar-value">
        {display}
        <span className="muted"> {rank ? `#${rank}` : "–"}</span>
      </span>
    </div>
  );
}

export function RetrievalPanel({
  hits,
  mode,
  citedNumbers,
  highlight,
}: {
  hits: Hit[];
  mode: Mode | null;
  citedNumbers: Map<string, number[]>;
  highlight: string | null;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const maxBm25 = Math.max(0, ...hits.map((h) => h.scores.bm25));
  const maxFused = Math.max(0, ...hits.map((h) => h.scores.fused));

  return (
    <aside className="panel" aria-label="How was this answered?">
      <h2>How was this answered?</h2>
      {!hits.length ? (
        <p className="muted">
          Ask a question to see which documentation sections were retrieved, with each retriever&apos;s score.
        </p>
      ) : (
        <>
          <p className="muted small">
            Top {hits.length} sections ranked by <strong>{mode ? MODE_LABELS[mode] : ""}</strong>, sent to
            Claude as citable documents. Every chunk also shows the other retrievers&apos; scores and ranks.
          </p>
          <ol className="hits">
            {hits.map((hit) => {
              const cited = citedNumbers.get(hit.chunk_id) ?? [];
              const expanded = open === hit.chunk_id;
              return (
                <li
                  key={hit.chunk_id}
                  id={`hit-${hit.chunk_id}`}
                  className={`hit ${cited.length ? "hit-cited" : ""} ${highlight === hit.chunk_id ? "hit-flash" : ""}`}
                >
                  <div className="hit-head">
                    <span className="rank">{hit.rank}</span>
                    <div className="hit-title">
                      {hit.url ? (
                        <a href={hit.url} target="_blank" rel="noreferrer">
                          {hit.heading_path.join(" › ")}
                        </a>
                      ) : (
                        <span>{hit.heading_path.join(" › ")}</span>
                      )}
                      <span className="muted small">{hit.source === "upload" ? hit.page : `/${hit.page}`}</span>
                    </div>
                    {cited.length > 0 && <span className="cited">cited {cited.map((n) => `[${n}]`).join("")}</span>}
                  </div>
                  <Bar label="BM25" value={hit.scores.bm25} max={maxBm25} display={hit.scores.bm25.toFixed(2)} rank={hit.scores.bm25_rank} />
                  <Bar label="Vector" value={Math.max(0, hit.scores.vector)} max={1} display={hit.scores.vector.toFixed(3)} rank={hit.scores.vector_rank} />
                  <Bar label="Fused" value={hit.scores.fused} max={maxFused} display={hit.scores.fused.toFixed(4)} rank={hit.scores.fused_rank} />
                  <button className="link small" onClick={() => setOpen(expanded ? null : hit.chunk_id)}>
                    {expanded ? "Hide text" : "Show chunk text"}
                  </button>
                  {expanded && <pre className="chunk-text">{hit.text}</pre>}
                </li>
              );
            })}
          </ol>
        </>
      )}
    </aside>
  );
}
