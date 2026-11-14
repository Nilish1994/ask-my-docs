// Typed client for the FastAPI backend (reached through the /api proxy).

export type Mode = "hybrid" | "bm25" | "vector";

export const MODE_LABELS: Record<Mode, string> = {
  hybrid: "Hybrid (RRF)",
  bm25: "BM25",
  vector: "Embeddings",
};

export interface Scores {
  bm25: number;
  bm25_rank: number | null;
  vector: number;
  vector_rank: number | null;
  fused: number;
  fused_rank: number | null;
}

export interface Hit {
  id: number;
  chunk_id: string;
  source: "fastapi" | "upload";
  page: string;
  page_title: string;
  heading_path: string[];
  url: string | null;
  text: string;
  rank: number;
  scores: Scores;
}

export interface Citation {
  document_index: number;
  chunk_id: string;
  page_title: string | null;
  heading: string;
  url: string | null;
  cited_text: string;
}

export interface Health {
  status: string;
  chunks: number;
  answering_enabled: boolean;
  model: string;
  limits: {
    questions_per_hour: number;
    questions_remaining_this_hour: number;
    daily_question_cap: number;
    daily_questions_used: number;
    max_question_chars: number;
    upload_max_files: number;
    upload_max_file_bytes: number;
    upload_ttl_seconds: number;
  };
}

export interface CollectionInfo {
  collection_id: string;
  files: { name: string; bytes: number; chunks: number }[];
  chunks: number;
  expires_at: number;
  expires_in_seconds: number;
}

export type AskEvent =
  | { type: "retrieval"; mode: Mode; results: Hit[] }
  | { type: "segment" }
  | { type: "text"; text: string }
  | ({ type: "citation" } & Citation)
  | { type: "fallback"; from: string; to: string }
  | { type: "done"; stop_reason: string; model: string; truncated: boolean; usage: Record<string, number> }
  | { type: "refusal"; category: string | null; message: string }
  | { type: "error"; code: string; message: string; retryable: boolean };

const BASE = "/api";

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* fall through */
  }
  return `Request failed (${response.status})`;
}

export async function getHealth(): Promise<Health> {
  const response = await fetch(`${BASE}/health`, { cache: "no-store" });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function search(q: string, mode: Mode, collectionId?: string, topK = 6): Promise<Hit[]> {
  const params = new URLSearchParams({ q, mode, top_k: String(topK) });
  if (collectionId) params.set("collection_id", collectionId);
  const response = await fetch(`${BASE}/search?${params}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).results;
}

export async function uploadFiles(files: File[]): Promise<CollectionInfo> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const response = await fetch(`${BASE}/collections`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function deleteCollection(id: string): Promise<void> {
  await fetch(`${BASE}/collections/${encodeURIComponent(id)}`, { method: "DELETE" });
}

/** POST /ask and yield parsed Server-Sent Events as they arrive. */
export async function* ask(
  question: string,
  mode: Mode,
  collectionId: string | undefined,
  signal: AbortSignal,
): AsyncGenerator<AskEvent> {
  const response = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ question, mode, collection_id: collectionId ?? null }),
    signal,
  });
  if (!response.ok || !response.body) {
    yield { type: "error", code: `http_${response.status}`, message: await errorMessage(response), retryable: false };
    return;
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (data) yield JSON.parse(data) as AskEvent;
    }
  }
}
