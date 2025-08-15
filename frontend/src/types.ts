export type ResultItem = {
  labels: Record<string, number>;
  label_probs: Record<string, number>;
  anomaly?: number | boolean;
  anomaly_score?: number;
};

export type PredictApiResult = {
  status: "ok" | "error" | "quota_exhausted";
  message?: string;
  results: Record<string, ResultItem>;
  used_thresholds: Record<string, number>;
  used_anomaly_threshold: number;
};
