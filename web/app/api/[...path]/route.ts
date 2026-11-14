// Same-origin proxy to the FastAPI service, so the browser never needs CORS
// and the API can stay bound to an internal address. Responses are passed
// through as streams, which keeps Server-Sent Events from /ask incremental.

import type { NextRequest } from "next/server";

const API_URL = process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8000";
const ALLOWED = new Set(["ask", "search", "collections", "health"]);
const FORWARD_REQUEST_HEADERS = ["content-type", "content-length", "accept", "x-forwarded-for"];
const FORWARD_RESPONSE_HEADERS = ["content-type", "retry-after", "cache-control"];

async function proxy(request: NextRequest, ctx: RouteContext<"/api/[...path]">) {
  const { path } = await ctx.params;
  if (!ALLOWED.has(path[0])) {
    return Response.json({ detail: "not found" }, { status: 404 });
  }

  const target = new URL(`${API_URL}/${path.map(encodeURIComponent).join("/")}`);
  target.search = request.nextUrl.search;

  const headers = new Headers();
  for (const name of FORWARD_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      // Required by Node's fetch when streaming a request body.
      ...(hasBody ? { duplex: "half" } : {}),
      cache: "no-store",
      signal: request.signal,
    } as RequestInit);
  } catch {
    return Response.json({ detail: "the API is unreachable" }, { status: 502 });
  }

  const responseHeaders = new Headers();
  for (const name of FORWARD_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  if (responseHeaders.get("content-type")?.startsWith("text/event-stream")) {
    responseHeaders.set("cache-control", "no-cache, no-transform");
    responseHeaders.set("x-accel-buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

export const GET = proxy;
export const POST = proxy;
export const DELETE = proxy;
