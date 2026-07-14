import type { ModelKey } from "./types";

export const MODEL_COPY: Record<ModelKey, { label: string; short: string; detail: string }> = {
  collaborative: {
    label: "People like you",
    short: "Best overall accuracy",
    detail: "Compares your ratings with similar taste patterns. Great once you have rated a few movies.",
  },
  content_based: {
    label: "Taste match",
    short: "Similar stories and genres",
    detail: "Looks at the genres and tags of movies you liked, then finds close matches.",
  },
  popularity: {
    label: "Crowd favorites",
    short: "Reliable starter picks",
    detail: "Uses trusted community favorites. Best before you have much personal history.",
  },
  clustering: {
    label: "Taste neighborhoods",
    short: "Groups your movie mood",
    detail: "Finds the taste neighborhoods you keep returning to and recommends from nearby favorites.",
  },
  neural: {
    label: "Taste embeddings",
    short: "Latent taste predictions",
    detail: "Learns hidden movie factors from rating behavior to estimate how well each movie fits your taste.",
  },
};

export const MODEL_ORDER: ModelKey[] = ["collaborative", "content_based", "popularity", "clustering", "neural"];
