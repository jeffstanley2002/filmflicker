import { Brain, Network, Sparkles, Target, TrendingUp } from "lucide-react";
import { MODEL_COPY, MODEL_ORDER } from "../lib/models";
import type { ModelKey } from "../lib/types";

const icons = {
  collaborative: Network,
  content_based: Target,
  popularity: TrendingUp,
  clustering: Sparkles,
  neural: Brain,
};

export function ModelSelector({ value, onChange }: { value: ModelKey; onChange: (model: ModelKey) => void }) {
  return (
    <div className="model-grid" role="radiogroup" aria-label="Recommendation model">
      {MODEL_ORDER.map((model) => {
        const Icon = icons[model];
        const copy = MODEL_COPY[model];
        return (
          <button key={model} className={value === model ? "model-option selected" : "model-option"} onClick={() => onChange(model)} role="radio" aria-checked={value === model}>
            <Icon size={20} />
            <span>
              <strong>{copy.label}</strong>
              <small>{copy.short}</small>
            </span>
          </button>
        );
      })}
    </div>
  );
}
