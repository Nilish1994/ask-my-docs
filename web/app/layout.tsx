import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ask My Docs: FastAPI Q&A with citations",
  description:
    "Ask questions about the FastAPI documentation. Answers stream from Claude with citations to the exact docs section, and every answer shows the BM25, embedding and hybrid retrieval scores behind it.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
