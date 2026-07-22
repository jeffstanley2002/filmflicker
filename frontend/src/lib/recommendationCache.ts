import type { ModelKey, Recommendation } from "./types";

const CACHE_PREFIX = "filmflicker:recommendations";
const TASTE_VERSION_PREFIX = "filmflicker:taste-version";
const FALLBACK_USER = "local-user";

type CachedRecommendations = {
  items: Recommendation[];
  savedAt: number;
  tasteVersion: number;
};

function storage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function decodeBase64Url(value: string) {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  return window.atob(padded);
}

export function recommendationUserKey(token: string) {
  const payload = token.split(".")[1];
  if (!payload) return FALLBACK_USER;
  try {
    const parsed = JSON.parse(decodeBase64Url(payload)) as { sub?: string };
    return parsed.sub ?? FALLBACK_USER;
  } catch {
    return FALLBACK_USER;
  }
}

function tasteVersionKey(token: string) {
  return `${TASTE_VERSION_PREFIX}:${recommendationUserKey(token)}`;
}

function cacheKey(token: string, model: ModelKey, n: number, tasteVersion: number) {
  return `${CACHE_PREFIX}:${recommendationUserKey(token)}:${tasteVersion}:${model}:${n}`;
}

export function getTasteVersion(token: string) {
  const value = storage()?.getItem(tasteVersionKey(token));
  return value ? Number(value) || 0 : 0;
}

export function bumpTasteVersion(token: string) {
  const next = getTasteVersion(token) + 1;
  storage()?.setItem(tasteVersionKey(token), String(next));
  window.dispatchEvent(new CustomEvent("filmflicker:taste-changed", { detail: { tasteVersion: next } }));
  return next;
}

export function readCachedRecommendations(token: string, model: ModelKey, n: number) {
  const tasteVersion = getTasteVersion(token);
  const raw = storage()?.getItem(cacheKey(token, model, n, tasteVersion));
  if (!raw) return null;
  try {
    const cached = JSON.parse(raw) as CachedRecommendations;
    return cached.tasteVersion === tasteVersion ? cached : null;
  } catch {
    return null;
  }
}

export function writeCachedRecommendations(token: string, model: ModelKey, n: number, items: Recommendation[]) {
  const tasteVersion = getTasteVersion(token);
  storage()?.setItem(
    cacheKey(token, model, n, tasteVersion),
    JSON.stringify({ items, savedAt: Date.now(), tasteVersion } satisfies CachedRecommendations),
  );
}
