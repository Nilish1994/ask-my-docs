"use client";

import { useEffect, useRef, useState } from "react";

import { AnswerView, numberSources, type Turn } from "@/components/Answer";
import { RetrievalPanel } from "@/components/RetrievalPanel";
import { UploadPanel } from "@/components/UploadPanel";
import type { CollectionInfo, Health, Mode } from "@/lib/api";
import { ask, getHealth, MODE_LABELS, search } from "@/lib/api";

const EXAMPLES = [
  "How do I make a query parameter optional?",
  "How do I open a database session per request and always close it?",
  "Where do I load an ML model once at startup?",
  "How do I enable CORS for my React frontend?",
];
const REPO_URL = "https://github.com/Nilish1994/ask-my-docs";

export function App() {
  const [tab, setTab] = useState<"ask" | "upload">("ask");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [scope, setScope] = useState<"docs" | "uploads">("docs");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [collection, setCollection] = useState<CollectionInfo | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const nextId = useRef(1);
  const turnsRef = useRef<HTMLDivElement | null>(null);

  const refreshHealth = () =>
    getHealth()
      .then((h) => {
        setHealth(h);
        setHealthError(false);
      })
      .catch(() => setHealthError(true));

  useEffect(() => {
    refreshHealth();
  }, []);

  // Shareable links: /?q=question&mode=bm25 runs the question once on load.
  const autoRan = useRef(false);
  useEffect(() => {
    if (autoRan.current || (!health && !healthError)) return;
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    if (!q) return;
    autoRan.current = true;
    const m = params.get("mode");
    const urlMode = m && m in MODE_LABELS ? (m as Mode) : mode;
    setMode(urlMode);
    (health?.answering_enabled ? submit : searchOnly)(q, urlMode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [health, healthError]);

  const busy = turns.some((t) => t.status === "retrieving" || t.status === "streaming");
  const current = turns.find((t) => t.id === selected) ?? turns[turns.length - 1];
  const maxChars = health?.limits.max_question_chars ?? 500;
  const collectionId = scope === "uploads" ? collection?.collection_id : undefined;

  function update(id: number, change: (turn: Turn) => Turn) {
    setTurns((all) => all.map((t) => (t.id === id ? change(t) : t)));
  }

  function newTurn(text: string, status: Turn["status"], turnMode: Mode): number {
    const id = nextId.current++;
    setTurns((all) => [
      ...all,
      { id, question: text, mode: turnMode, scope: collectionId ? "uploads" : "docs", status, segments: [], hits: [] },
    ]);
    setSelected(id);
    // Scroll only the conversation column, never the page.
    setTimeout(() => turnsRef.current?.scrollTo({ top: turnsRef.current.scrollHeight, behavior: "smooth" }), 50);
    return id;
  }

  async function submit(text = question, turnMode: Mode = mode) {
    text = text.trim();
    if (!text || busy) return;
    setQuestion("");
    const id = newTurn(text, "retrieving", turnMode);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      for await (const event of ask(text, turnMode, collectionId, controller.signal)) {
        switch (event.type) {
          case "retrieval":
            update(id, (t) => ({ ...t, hits: event.results, status: "streaming" }));
            break;
          case "segment":
            update(id, (t) => ({ ...t, segments: [...t.segments, { text: "", citations: [] }] }));
            break;
          case "text":
            update(id, (t) => {
              const segments = t.segments.length ? [...t.segments] : [{ text: "", citations: [] }];
              const last = segments[segments.length - 1];
              segments[segments.length - 1] = { ...last, text: last.text + event.text };
              return { ...t, segments };
            });
            break;
          case "citation":
            update(id, (t) => {
              const segments = t.segments.length ? [...t.segments] : [{ text: "", citations: [] }];
              const last = segments[segments.length - 1];
              segments[segments.length - 1] = { ...last, citations: [...last.citations, event] };
              return { ...t, segments };
            });
            break;
          case "fallback":
            update(id, (t) => ({ ...t, fallback: `${event.from} declined part of this request; ${event.to} continued.` }));
            break;
          case "done":
            update(id, (t) => ({ ...t, status: "done", model: event.model, truncated: event.truncated }));
            break;
          case "refusal":
            update(id, (t) => ({ ...t, status: "refused", segments: [], error: event.message }));
            break;
          case "error":
            update(id, (t) => ({ ...t, status: "error", error: event.message }));
            break;
        }
      }
      update(id, (t) => (t.status === "streaming" ? { ...t, status: "done" } : t));
    } catch (e) {
      if (!controller.signal.aborted) {
        update(id, (t) => ({ ...t, status: "error", error: e instanceof Error ? e.message : String(e) }));
      } else {
        update(id, (t) => ({ ...t, status: "done" }));
      }
    } finally {
      refreshHealth();
    }
  }

  async function searchOnly(text = question, turnMode: Mode = mode) {
    text = text.trim();
    if (!text || busy) return;
    setQuestion("");
    const id = newTurn(text, "retrieving", turnMode);
    try {
      const hits = await search(text, turnMode, collectionId);
      update(id, (t) => ({ ...t, hits, status: "search-only" }));
    } catch (e) {
      update(id, (t) => ({ ...t, status: "error", error: e instanceof Error ? e.message : String(e) }));
    }
  }

  function showChunk(chunkId: string) {
    setHighlight(chunkId);
    document.getElementById(`hit-${chunkId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => setHighlight(null), 1600);
  }

  const citedNumbers = new Map<string, number[]>();
  if (current) {
    for (const { number, citation } of numberSources(current).values()) {
      citedNumbers.set(citation.chunk_id, [...(citedNumbers.get(citation.chunk_id) ?? []), number]);
    }
  }

  const limits = health?.limits;
  const answering = health?.answering_enabled ?? true;

  return (
    <div className="shell">
      <header className="header">
        <div>
          <h1>Ask My Docs</h1>
          <p className="muted">
            Questions about the FastAPI docs, answered by Claude with citations that link to the exact section.
          </p>
        </div>
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "ask"} onClick={() => setTab("ask")}>
            Ask
          </button>
          <button role="tab" aria-selected={tab === "upload"} onClick={() => setTab("upload")}>
            Upload{collection ? " ·1" : ""}
          </button>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
      </header>

      {healthError && <p className="error banner">The API is not reachable right now.</p>}
      {health && !answering && (
        <p className="notice banner">
          Answering is disabled on this server (no API key). Search still works and shows the retrieved sections.
        </p>
      )}

      {tab === "upload" ? (
        <UploadPanel
          health={health}
          collection={collection}
          onCollection={(c) => {
            setCollection(c);
            setScope(c ? "uploads" : "docs");
            if (c) setTab("ask");
          }}
        />
      ) : (
        <main className="layout">
          <section className="chat">
            <div className="controls">
              <div className="segmented" role="radiogroup" aria-label="Retrieval mode">
                {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
                  <button key={m} role="radio" aria-checked={mode === m} onClick={() => setMode(m)}>
                    {MODE_LABELS[m]}
                  </button>
                ))}
              </div>
              {collection && (
                <div className="segmented" role="radiogroup" aria-label="Corpus">
                  <button role="radio" aria-checked={scope === "docs"} onClick={() => setScope("docs")}>
                    FastAPI docs
                  </button>
                  <button role="radio" aria-checked={scope === "uploads"} onClick={() => setScope("uploads")}>
                    Your uploads
                  </button>
                </div>
              )}
            </div>

            <div className="turns" ref={turnsRef}>
              {turns.length === 0 && (
                <div className="empty">
                  <p className="muted">Try one of these:</p>
                  <div className="examples">
                    {EXAMPLES.map((example) => (
                      <button
                        key={example}
                        onClick={() => (answering ? submit(example) : searchOnly(example))}
                        disabled={busy}
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {turns.map((turn) => (
                <AnswerView
                  key={turn.id}
                  turn={turn}
                  selected={current?.id === turn.id}
                  onSelect={() => setSelected(turn.id)}
                  onCitationClick={showChunk}
                />
              ))}
            </div>

            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <textarea
                value={question}
                maxLength={maxChars}
                rows={2}
                placeholder={scope === "uploads" ? "Ask about your uploaded files…" : "Ask a question about FastAPI…"}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit();
                  }
                }}
              />
              <div className="composer-row">
                <span className="muted small">
                  {question.length}/{maxChars}
                  {limits && answering && ` · ${limits.questions_remaining_this_hour} of ${limits.questions_per_hour} questions left this hour`}
                </span>
                <div className="row">
                  {busy ? (
                    <button type="button" onClick={() => abortRef.current?.abort()}>
                      Stop
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => searchOnly()}
                      disabled={!question.trim()}
                      title="Retrieval only: no LLM call"
                    >
                      Search only
                    </button>
                  )}
                  <button className="primary" type="submit" disabled={!question.trim() || busy || !answering}>
                    Ask
                  </button>
                </div>
              </div>
            </form>
          </section>

          <RetrievalPanel
            hits={current?.hits ?? []}
            mode={current?.mode ?? null}
            citedNumbers={citedNumbers}
            highlight={highlight}
          />
        </main>
      )}

      <footer className="footer muted small">
        Docs content © Sebastián Ramírez, MIT licence. Retrieval: BM25 + bge-small-en-v1.5 embeddings, fused with RRF.
        Answers: {health?.model ?? "Claude"}.
      </footer>
    </div>
  );
}
