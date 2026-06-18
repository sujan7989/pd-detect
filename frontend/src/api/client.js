import axios from "axios";

// In production (Vercel), set VITE_API_URL to your Render backend URL
// e.g. https://pd-detect-api.onrender.com
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE, timeout: 90000 });

api.interceptors.response.use(
  (r) => r,
  (err) => {
    const isNet =
      err.code === "ERR_NETWORK" ||
      err.code === "ECONNREFUSED" ||
      err.message?.toLowerCase().includes("network");
    err.userMessage = isNet
      ? "Cannot reach the API server. Please try again in a moment."
      : err.response?.data?.detail || err.message || "An unexpected error occurred.";
    return Promise.reject(err);
  }
);

export const checkHealth        = ()        => api.get("/health").then((r) => r.data);
export const getHistory         = ()        => api.get("/history").then((r) => r.data);
export const clearHistory       = ()        => api.delete("/history").then((r) => r.data);
export const deleteHistoryEntry = (id)      => api.delete(`/history/${id}`).then((r) => r.data);
export const getStats           = ()        => api.get("/stats").then((r) => r.data);

export const analyzeAudio = (blob, filename, onProgress) => {
  const form = new FormData();
  form.append("file", blob, filename);
  return api
    .post("/analyze", form, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
      },
    })
    .then((r) => r.data);
};

export const downloadReport = async (result) => {
  const response = await api.post("/report", result, { responseType: "blob" });
  const url = URL.createObjectURL(
    new Blob([response.data], { type: "application/pdf" })
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `pd_report_${result.id || Date.now()}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export default api;
