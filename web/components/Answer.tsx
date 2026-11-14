"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { Citation, Hit, Mode } from "@/lib/api";
import { MODE_LABELS } from "@/lib/api";

export interface Segment {
  text: string;
  citations: Citation[];
}

export interface Turn {
  id: number;
  question: string;
  mode: Mode;
  scope: "docs" | "uploads";
  status: "retrieving" | "streaming" | "done" | "error" | "refused" | "search-only";
  segments: Segment[];
  hits: Hit[];
  error?: string;
  fallback?: string;
  model?: string;
  truncated?: boolean;
}

export interface NumberedSource {
  number: number;
  citation: Citation;
}

const citationKey = (c: Citation) => `${c.chunk_id}|${c.url ?? c.heading}`;

/** Number each distinct cited section in order of first appearance. */
export function numberSources(turn: Turn): Map<string, NumberedSource> {
  const sources = new Map<string, NumberedSource>();
  for (const segment of turn.segments) {
    for (const citation of segment.citations) {
      const key = citationKey(citation);
      if (!sources.has(key)) sources.set(key, { number: sources.size + 1, citation });
    }
  }
  return sources;
}

/**
 * Join streamed segments into one Markdown string, appending a chip marker
 * (a #cite-N link) after each cited segment. Markers inside an open code
 * fence are deferred until the fence closes so code blocks stay intact.
 */
function toMarkdown(turn: Turn, sources: Map<string, NumberedSource>): string {
  let out = "";
  let pending: number[] = [];
  for (const segment of turn.segments) {
    out += segment.text;
    for (const citation of segment.citations) {
      const number = sources.get(citationKey(citation))!.number;
      if (!pending.includes(number)) pending.push(number);
    }
    const insideFence = (out.match(/```/g)?.length ?? 0) % 2 === 1;
    if (pending.length && !insideFence) {
      const markers = pending.map((n) => `[${n}](#cite-${n})`).join("");
      out = out.replace(/(\s*)$/, ` ${markers}$1`);
      pending = [];
    }
  }
  return out;
}

export function AnswerView({
  turn,
  selected,
  onSelect,
  onCitationClick,
}: {
  turn: Turn;
  selected: boolean;
  onSelect: () => void;
  onCitationClick: (chunkId: string) => void;
}) {
  const sources = numberSources(turn);
  const byNumber = new Map([...sources.values()].map((s) => [s.number, s.citation]));
  const markdown = toMarkdown(turn, sources);

  return (
    <article className={`turn ${selected ? "turn-selected" : ""}`} onClick={onSelect}>
      <div className="question">
        <span>{turn.question}</span>
        <span className="tag">
          {turn.scope === "uploads" ? "your uploads · " : ""}
          {MODE_LABELS[turn.mode]}
        </span>
      </div>

      <div className="answer">
        {turn.status === "retrieving" && <p className="muted pulse">Retrieving relevant sections…</p>}
        {turn.status === "search-only" && (
          <p className="muted">Retrieval only, no LLM call. The ranked sections are in the panel.</p>
        )}
        {turn.fallback && <p className="notice">{turn.fallback}</p>}
        {turn.status === "refused" ? (
          <p className="error">{turn.error}</p>
        ) : (
          markdown && (
            <div className="markdown">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ href, children }) => {
                    const match = href?.match(/^#cite-(\d+)$/);
                    if (match) {
                      const citation = byNumber.get(Number(match[1]));
                      if (!citation) return null;
                      return (
                        <CitationChip
                          number={Number(match[1])}
                          citation={citation}
                          onClick={() => onCitationClick(citation.chunk_id)}
                        />
                      );
                    }
                    return (
                      <a href={href} target="_blank" rel="noreferrer">
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {markdown}
              </ReactMarkdown>
            </div>
          )
        )}
        {turn.status === "streaming" && <span className="cursor" aria-hidden />}
        {turn.status === "error" && <p className="error">{turn.error}</p>}
        {turn.truncated && <p className="notice">The answer hit the length cap and was cut short.</p>}
      </div>

      {sources.size > 0 && turn.status !== "refused" && (
        <ol className="sources">
          {[...sources.values()].map(({ number, citation }) => (
            <li key={number}>
              <span className="chip static">{number}</span>
              {citation.url ? (
                <a href={citation.url} target="_blank" rel="noreferrer">
                  {citation.page_title} › {citation.heading}
                </a>
              ) : (
                <button className="link" onClick={() => onCitationClick(citation.chunk_id)}>
                  {citation.page_title} › {citation.heading}
                </button>
              )}
            </li>
          ))}
        </ol>
      )}
      {turn.model && turn.status === "done" && <p className="meta">Answered by {turn.model}</p>}
    </article>
  );
}

function CitationChip({ number, citation, onClick }: { number: number; citation: Citation; onClick: () => void }) {
  const title = `${citation.page_title ?? ""} › ${citation.heading}\n“${citation.cited_text.trim().slice(0, 200)}”`;
  if (citation.url) {
    return (
      <a className="chip" href={citation.url} target="_blank" rel="noreferrer" title={title}>
        {number}
      </a>
    );
  }
  return (
    <button className="chip" title={title} onClick={onClick}>
      {number}
    </button>
  );
}
