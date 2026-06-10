"use client";

import { FormEvent, useState } from "react";

import { predictRaces, type PredictRequest, type PredictResponse } from "@/lib/api";
import { PredictionsTable } from "@/components/PredictionsTable";

type ManualConditionsFormProps = {
  raceId?: number;
  defaultElevGainFt?: number | null;
  defaultElevLossFt?: number | null;
  defaultTempF?: number | null;
  defaultAsOf?: string;
  maxAsOf?: string;
  title?: string;
};

export function ManualConditionsForm({
  raceId,
  defaultElevGainFt,
  defaultElevLossFt,
  defaultTempF,
  defaultAsOf = "",
  maxAsOf,
  title = "Manual course conditions",
}: ManualConditionsFormProps) {
  const [elevGainFt, setElevGainFt] = useState(
    defaultElevGainFt != null ? String(defaultElevGainFt) : "0"
  );
  const [elevLossFt, setElevLossFt] = useState(
    defaultElevLossFt != null ? String(defaultElevLossFt) : "0"
  );
  const [tempF, setTempF] = useState(
    defaultTempF != null ? String(defaultTempF) : ""
  );
  const [asOf, setAsOf] = useState(defaultAsOf);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const body: PredictRequest = {
        elev_gain_ft: Number(elevGainFt),
        elev_loss_ft: Number(elevLossFt),
      };
      if (raceId != null) body.race_id = raceId;
      if (tempF.trim()) body.temp_f = Number(tempF);
      if (asOf.trim()) body.as_of = asOf;

      const response = await predictRaces(body);
      setResult(response);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-medium">{title}</h2>
        <p className="text-sm text-zinc-400">
          Override elevation or temperature when course data is incomplete.
        </p>
      </div>

      <form
        onSubmit={onSubmit}
        className="grid gap-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 md:grid-cols-2"
      >
        <label className="flex flex-col gap-1 text-sm">
          Elevation gain (ft)
          <input
            type="number"
            min="0"
            step="1"
            required
            value={elevGainFt}
            onChange={(e) => setElevGainFt(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          Elevation loss (ft)
          <input
            type="number"
            min="0"
            step="1"
            required
            value={elevLossFt}
            onChange={(e) => setElevLossFt(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          Temperature (°F)
          <input
            type="number"
            min="-20"
            max="120"
            step="0.1"
            value={tempF}
            onChange={(e) => setTempF(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          As-of date
          <input
            type="date"
            value={asOf}
            max={maxAsOf}
            onChange={(e) => setAsOf(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          />
        </label>

        <div className="md:col-span-2">
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-500 disabled:opacity-60"
          >
            {loading ? "Predicting…" : "Predict all distances"}
          </button>
        </div>
      </form>

      {error && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-red-200">
          {error}
        </div>
      )}

      {result && <PredictionsTable result={result} />}
    </div>
  );
}
