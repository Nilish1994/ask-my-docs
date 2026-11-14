"use client";

import { useEffect, useState } from "react";

import type { CollectionInfo, Health } from "@/lib/api";
import { deleteCollection, uploadFiles } from "@/lib/api";

const ACCEPT = ".md,.markdown,.txt,.pdf";

function formatRemaining(seconds: number): string {
  const minutes = Math.max(0, Math.floor(seconds / 60));
  return minutes >= 1 ? `${minutes} min` : "under a minute";
}

export function UploadPanel({
  health,
  collection,
  onCollection,
}: {
  health: Health | null;
  collection: CollectionInfo | null;
  onCollection: (collection: CollectionInfo | null) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now() / 1000), 15_000);
    return () => clearInterval(timer);
  }, []);

  const maxFiles = health?.limits.upload_max_files ?? 5;
  const maxBytes = health?.limits.upload_max_file_bytes ?? 2 * 1024 * 1024;
  const ttlMinutes = Math.round((health?.limits.upload_ttl_seconds ?? 3600) / 60);

  function choose(list: FileList | null) {
    setError(null);
    const chosen = Array.from(list ?? []);
    if (chosen.length > maxFiles) return setError(`Choose at most ${maxFiles} files.`);
    const tooBig = chosen.find((f) => f.size > maxBytes);
    if (tooBig) return setError(`${tooBig.name} is larger than ${maxBytes / (1024 * 1024)} MB.`);
    setFiles(chosen);
  }

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      if (collection) await deleteCollection(collection.collection_id);
      onCollection(await uploadFiles(files));
      setFiles([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!collection) return;
    await deleteCollection(collection.collection_id);
    onCollection(null);
  }

  const expired = collection ? collection.expires_at <= now : false;

  return (
    <section className="upload">
      <h2>Ask your own documents</h2>
      <p className="muted">
        Upload up to {maxFiles} Markdown, text or PDF files ({maxBytes / (1024 * 1024)} MB each). They are parsed and
        indexed in memory only, never written to disk, and deleted after {ttlMinutes} minutes.
      </p>

      <label className="dropzone">
        <input type="file" multiple accept={ACCEPT} onChange={(e) => choose(e.target.files)} />
        <span>{files.length ? files.map((f) => f.name).join(", ") : "Choose .md, .txt or .pdf files"}</span>
      </label>

      <div className="row">
        <button className="primary" disabled={!files.length || busy} onClick={submit}>
          {busy ? "Indexing…" : "Upload and index"}
        </button>
        {error && <span className="error">{error}</span>}
      </div>

      {collection && !expired && (
        <div className="collection">
          <div>
            <strong>Active collection</strong> · {collection.chunks} chunks · expires in{" "}
            {formatRemaining(collection.expires_at - now)}
          </div>
          <ul>
            {collection.files.map((f) => (
              <li key={f.name}>
                {f.name} <span className="muted">({f.chunks} chunks)</span>
              </li>
            ))}
          </ul>
          <button className="link" onClick={remove}>
            Delete now
          </button>
        </div>
      )}
      {collection && expired && <p className="notice">Your uploaded collection has expired.</p>}
    </section>
  );
}
