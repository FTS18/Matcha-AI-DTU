/** HTTP API client for the Matcha AI orchestrator.
 *  Import this in apps/web and apps/mobile — pass the base URL from your env/constants.
 */

import type { MatchSummary, MatchDetail } from "./types";
import { fetchWithRetry } from "./utils";

const getAuthHeaders = (): Record<string, string> => {
  if (typeof window === 'undefined') return {};
  const token = localStorage.getItem('auth_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

export function createApiClient(baseUrl: string) {
  // Strip trailing slash if present for consistent concatenation
  const cleanBase = baseUrl.replace(/\/$/, "");
  const apiBase = `${cleanBase}/api/v1`;

  return {
    /** Helper to resolve relative asset paths (e.g. /uploads/...) to full URLs */
    getAssetUrl: (path: string | null): string => {
      if (!path) return "";
      if (path.startsWith("http")) return path;
      return `${cleanBase}${path}`;
    },

    getMatches: (): Promise<MatchSummary[]> =>
      fetchWithRetry(`${apiBase}/matches`, { headers: getAuthHeaders() }).then((r) => r.json()),

    getMatch: (id: string): Promise<MatchDetail> =>
      fetchWithRetry(`${apiBase}/matches/${id}`, { headers: getAuthHeaders() }).then((r) => r.json()),

    deleteMatch: (id: string): Promise<Response> =>
      fetchWithRetry(`${apiBase}/matches/${id}`, { method: "DELETE", headers: getAuthHeaders() }),

    reanalyze: (id: string): Promise<Response> =>
      fetchWithRetry(`${apiBase}/matches/${id}/reanalyze`, { method: "POST", headers: getAuthHeaders() }),

    generateReel: (id: string, aspectRatio: "16:9" | "9:16"): Promise<Response> =>
      fetchWithRetry(`${apiBase}/matches/${id}/reanalyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ aspect_ratio: aspectRatio }),
      }),

    uploadVideo: (file: File | Blob, onProgress?: (pct: number) => void): Promise<MatchSummary> =>
      new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const form = new FormData();
        form.append("file", file);
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
          else reject(new Error(`Upload failed: ${xhr.status}`));
        };
        xhr.onerror = () => reject(new Error("Network error"));
        xhr.open("POST", `${apiBase}/matches/upload`);
        const authHeaders = getAuthHeaders();
        if (authHeaders.Authorization) {
          xhr.setRequestHeader("Authorization", authHeaders.Authorization);
        }
        xhr.send(form);
      }),

    uploadYoutube: (url: string, startTime?: number, endTime?: number): Promise<MatchSummary> =>
      fetchWithRetry(`${apiBase}/matches/youtube`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ url, start_time: startTime, end_time: endTime }),
      }).then((r) => {
        if (!r.ok) throw new Error(`YouTube upload failed: ${r.statusText}`);
        return r.json();
      }),

    getYtInfo: (url: string): Promise<{ title: string; duration: number; thumbnail: string; channel: string }> =>
      fetchWithRetry(`${apiBase}/matches/yt-info?url=${encodeURIComponent(url)}`, {
        headers: getAuthHeaders(),
      }).then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      }),

    login: (body: any): Promise<{ access_token: string; user: any }> =>
      fetchWithRetry(`${cleanBase}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      }),

    register: (body: any): Promise<{ access_token: string; user: any }> =>
      fetchWithRetry(`${cleanBase}/api/v1/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      }),

    getMe: (): Promise<any> =>
      fetchWithRetry(`${apiBase}/auth/me`, {
        method: "GET",
        headers: getAuthHeaders(),
      }).then(async (r) => {
        if (!r.ok) {
          const err = new Error("Unauthorized") as any;
          err.status = r.status;
          throw err;
        }
        return r.json();
      }).catch((err) => {
        // Re-throw with status info so AuthContext can decide what to do
        throw err;
      }),
  };
}


export type ApiClient = ReturnType<typeof createApiClient>;
