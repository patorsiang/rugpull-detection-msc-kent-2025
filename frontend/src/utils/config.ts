// Prefer runtime config if present, else fall back to Vite env, else default.
declare global {
  interface Window {
    __APP_CONFIG__?: { CURL_ENDPOINT?: string };
  }
}

export function getCurlEndpoint(): string {
  return (
    window.__APP_CONFIG__?.CURL_ENDPOINT ||
    (import.meta as ImportMeta).env?.VITE_CURL_ENDPOINT ||
    "http://localhost:8000/api/predict"
  );
}
