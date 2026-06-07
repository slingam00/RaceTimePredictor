export type PredictionItem = {
  distance_label: string;
  predicted_time_sec: number;
  pace_min_per_mi: number;
};

export type PredictResponse = {
  as_of: string;
  elev_gain_ft: number;
  elev_loss_ft: number;
  temp_f: number;
  temp_source: string;
  predictions: PredictionItem[];
};

export type PredictRequest = {
  elev_gain_ft: number;
  elev_loss_ft: number;
  temp_f?: number;
  as_of?: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function predictRaces(body: PredictRequest): Promise<PredictResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail =
        typeof payload.detail === "string"
          ? payload.detail
          : "Prediction request failed";
      throw new Error(detail);
    }
    return payload as PredictResponse;
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error(
        `Cannot reach API at ${API_BASE}. Start it with: uvicorn api.main:app --reload --port 8000`
      );
    }
    throw err;
  }
}
