export type PredictionItem = {
  distance_label: string;
  predicted_time_sec: number;
  pace_min_per_mi: number;
  interval_low_sec: number;
  interval_high_sec: number;
  confidence: number;
};

export type PredictResponse = {
  as_of: string;
  elev_gain_ft: number;
  elev_loss_ft: number;
  temp_f: number;
  temp_source: string;
  race_id?: number | null;
  race_name?: string | null;
  elev_source?: string | null;
  warnings: string[];
  predictions: PredictionItem[];
};

export type PredictRequest = {
  race_id?: number;
  event_id?: number;
  elev_gain_ft?: number;
  elev_loss_ft?: number;
  temp_f?: number;
  as_of?: string;
};

export type RaceSummary = {
  race_id: number;
  name: string;
  city?: string | null;
  state?: string | null;
  next_date?: string | null;
  offered_distances: string[];
};

export type RaceSearchResponse = {
  races: RaceSummary[];
  page: number;
  results_per_page: number;
};

export type RaceEventDetail = {
  event_id: number;
  name: string;
  distance_label: string;
  distance_mi: number;
};

export type RaceDetail = {
  race_id: number;
  name: string;
  city?: string | null;
  state?: string | null;
  race_date: string;
  elev_gain_ft?: number | null;
  elev_loss_ft?: number | null;
  elev_source?: string | null;
  temp_f?: number | null;
  weather_source?: string | null;
  offered_events: RaceEventDetail[];
  warnings: string[];
};

export type SearchRacesParams = {
  q?: string;
  city?: string;
  state?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  results_per_page?: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_BASE}${path}`, init);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail =
        typeof payload.detail === "string"
          ? payload.detail
          : "Request failed";
      throw new Error(detail);
    }
    return payload as T;
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error(
        `Cannot reach API at ${API_BASE}. Start it with: uvicorn api.main:app --reload --port 8000`
      );
    }
    throw err;
  }
}

export async function searchRaces(
  params: SearchRacesParams = {}
): Promise<RaceSearchResponse> {
  const query = new URLSearchParams();
  if (params.q) query.set("q", params.q);
  if (params.city) query.set("city", params.city);
  if (params.state) query.set("state", params.state);
  if (params.start_date) query.set("start_date", params.start_date);
  if (params.end_date) query.set("end_date", params.end_date);
  if (params.page) query.set("page", String(params.page));
  if (params.results_per_page) {
    query.set("results_per_page", String(params.results_per_page));
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiFetch<RaceSearchResponse>(`/api/races/search${suffix}`);
}

export async function getRace(raceId: number): Promise<RaceDetail> {
  return apiFetch<RaceDetail>(`/api/races/${raceId}`);
}

export async function predictRaces(body: PredictRequest): Promise<PredictResponse> {
  return apiFetch<PredictResponse>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
