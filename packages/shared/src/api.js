/** HTTP API client for the Matcha AI orchestrator.
 *  Import this in apps/web and apps/mobile — pass the base URL from your env/constants.
 */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import { fetchWithRetry } from "./utils";
const getAuthHeaders = () => {
    if (typeof window === 'undefined')
        return {};
    const token = localStorage.getItem('auth_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
};
export function createApiClient(baseUrl) {
    // Strip trailing slash if present for consistent concatenation
    const cleanBase = baseUrl.replace(/\/$/, "");
    const apiBase = `${cleanBase}/api/v1`;
    return {
        /** Helper to resolve relative asset paths (e.g. /uploads/...) to full URLs */
        getAssetUrl: (path) => {
            if (!path)
                return "";
            if (path.startsWith("http"))
                return path;
            return `${cleanBase}${path}`;
        },
        getMatches: () => fetchWithRetry(`${apiBase}/matches`, { headers: getAuthHeaders() }).then((r) => r.json()),
        getMatch: (id) => fetchWithRetry(`${apiBase}/matches/${id}`, { headers: getAuthHeaders() }).then((r) => r.json()),
        deleteMatch: (id) => fetchWithRetry(`${apiBase}/matches/${id}`, { method: "DELETE", headers: getAuthHeaders() }),
        reanalyze: (id) => fetchWithRetry(`${apiBase}/matches/${id}/reanalyze`, { method: "POST", headers: getAuthHeaders() }),
        uploadVideo: (file, onProgress) => new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            const form = new FormData();
            form.append("file", file);
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable && onProgress)
                    onProgress(Math.round((e.loaded / e.total) * 100));
            };
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300)
                    resolve(JSON.parse(xhr.responseText));
                else
                    reject(new Error(`Upload failed: ${xhr.status}`));
            };
            xhr.onerror = () => reject(new Error("Network error"));
            xhr.open("POST", `${apiBase}/matches/upload`);
            const authHeaders = getAuthHeaders();
            if (authHeaders.Authorization) {
                xhr.setRequestHeader("Authorization", authHeaders.Authorization);
            }
            xhr.send(form);
        }),
        uploadYoutube: (url, startTime, endTime) => fetchWithRetry(`${apiBase}/matches/youtube`, {
            method: "POST",
            headers: Object.assign({ "Content-Type": "application/json" }, getAuthHeaders()),
            body: JSON.stringify({ url, start_time: startTime, end_time: endTime }),
        }).then((r) => {
            if (!r.ok)
                throw new Error(`YouTube upload failed: ${r.statusText}`);
            return r.json();
        }),
        getYtInfo: (url) => fetchWithRetry(`${apiBase}/matches/yt-info?url=${encodeURIComponent(url)}`, {
            headers: getAuthHeaders(),
        }).then((r) => __awaiter(this, void 0, void 0, function* () {
            if (!r.ok)
                throw new Error(yield r.text());
            return r.json();
        })),
        login: (body) => fetchWithRetry(`${cleanBase}/api/v1/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        }).then((r) => __awaiter(this, void 0, void 0, function* () {
            if (!r.ok)
                throw new Error(yield r.text());
            return r.json();
        })),
        register: (body) => fetchWithRetry(`${cleanBase}/api/v1/auth/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        }).then((r) => __awaiter(this, void 0, void 0, function* () {
            if (!r.ok)
                throw new Error(yield r.text());
            return r.json();
        })),
        getMe: () => fetchWithRetry(`${apiBase}/auth/me`, {
            method: "GET",
            headers: getAuthHeaders(),
        }).then((r) => __awaiter(this, void 0, void 0, function* () {
            if (!r.ok) {
                if (r.status === 401 && typeof window !== 'undefined') {
                    localStorage.removeItem('auth_token');
                }
                throw new Error("Unauthorized");
            }
            return r.json();
        })),
    };
}
