const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

export function getHealth() {
  return request("/api/health");
}

export function getQuestions() {
  return request("/api/questions");
}

export function analyzeEmotion(payload) {
  return request("/api/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function retrainModel() {
  return request("/api/train", {
    method: "POST",
    body: JSON.stringify({}),
  });
}
