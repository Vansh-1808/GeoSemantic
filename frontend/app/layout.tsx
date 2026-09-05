import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { AppShell } from "@/components/layout/AppShell";

export const metadata: Metadata = {
  title: "GeoSemantic — Satellite Intelligence Platform",
  description:
    "Semantic retrieval and multi-temporal change analysis of satellite imagery. SIH 2026.",
  keywords: ["satellite", "geospatial", "change detection", "semantic search", "remote sensing"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-950 text-gray-100 antialiased">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
