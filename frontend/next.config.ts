import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Analyst dashboard — desktop first, no need for image optimization CDN
  images: {
    unoptimized: true, // Serve thumbnails directly from FastAPI
  },
  // Forward /api/* to the FastAPI backend during development
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/thumbnails/:path*",
        destination: "http://localhost:8000/thumbnails/:path*",
      },
      {
        source: "/previews/:path*",
        destination: "http://localhost:8000/previews/:path*",
      },
      {
        source: "/scenes/:path*",
        destination: "http://localhost:8000/scenes/:path*",
      },
      {
        source: "/tiles/:path*",
        destination: "http://localhost:8000/tiles/:path*",
      },
      {
        source: "/models/:path*",
        destination: "http://localhost:8000/models/:path*",
      },
      {
        source: "/embedding/:path*",
        destination: "http://localhost:8000/embedding/:path*",
      },
      {
        source: "/vector/:path*",
        destination: "http://localhost:8000/vector/:path*",
      },
      {
        source: "/exports/:path*",
        destination: "http://localhost:8000/exports/:path*",
      },
    ];
  },
};

export default nextConfig;
