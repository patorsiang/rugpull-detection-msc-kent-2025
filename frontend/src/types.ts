export type PredictApiResult = {
  status: string;
  results: Record<
    string,
    {
      labels: Record<string, number>;
      label_probs: Record<string, number>;
      anomaly?: number | boolean;
      anomaly_score?: number;
    }
  >;
};
