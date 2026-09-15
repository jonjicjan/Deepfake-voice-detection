const API_KEY_STORAGE = 'voiceshield_api_key';

export function getApiKey() {
  return sessionStorage.getItem(API_KEY_STORAGE) || '';
}

export function setApiKey(key) {
  sessionStorage.setItem(API_KEY_STORAGE, key);
}

export function clearApiKey() {
  sessionStorage.removeItem(API_KEY_STORAGE);
}

export function authHeaders(extra = {}) {
  const headers = { ...extra };
  const key = getApiKey();
  if (key) headers['X-API-Key'] = key;
  return headers;
}

export async function apiFetch(path, options = {}) {
  const headers = authHeaders(options.headers || {});
  const res = await fetch(`/api${path}`, { ...options, headers });
  if (res.status === 401) {
    clearApiKey();
    throw new Error('Authentication required. Please sign in with a valid API key.');
  }
  if (res.status === 403) {
    throw new Error('Access denied. Administrator privileges required.');
  }
  return res;
}

export async function checkHealth() {
  const res = await fetch('/health');
  return res.ok;
}
