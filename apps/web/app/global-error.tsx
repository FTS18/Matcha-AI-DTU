"use client";

import { useEffect } from "react";

export default function GlobalError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error("Global error caught:", error);
  }, [error]);

  return (
    <html lang="en">
      <body>
        <div
          style={{
            padding: "2rem",
            fontFamily: "sans-serif",
            textAlign: "center",
          }}
        >
          <h2>Application Error</h2>
          <p>A fatal error occurred in the Match Intelligence pipeline.</p>
          <button onClick={() => reset()} style={{ padding: "0.5rem 1rem", marginTop: "1rem" }}>
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
