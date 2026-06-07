"use client";

import { FormEvent, useState } from "react";

import { predictRaces, type PredictResponse } from "@/lib/api";
import { formatPace, formatTime } from "@/lib/format";

export default function HomePage() {
  const [elevGainFt, setElevGainFt] = useState("0");
  const [elevLossFt, setElevLossFt] = useState("0");
  const [tempF, setTempF] = useState("72");
  const [asOf, setAsOf] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const body = {
        elev_gain_ft: Number(elevGainFt),
        elev_loss_ft: Number(elevLossFt),
        temp_f: tempF.trim() ? Number(tempF) : undefined,
        as_of: asOf.trim() || undefined,
      };
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
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col gap-8 px-6 py-12">
      <header className="space-y-2">
        <p className="text-sm uppercase tracking-wide text-zinc-400">Race Time Predictor</p>
        <h1 className="text-3xl font-semibold">Manual race conditions</h1>
        <p className="text-zinc-400">
          Enter course elevation and temperature. Requires FastAPI on{" "}
          <code className="rounded bg-zinc-900 px-1.5 py-0.5 text-zinc-200">
            {process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}
          </code>
          .
        </p>
      </header>

      <form
        onSubmit={onSubmit}
        className="grid gap-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 md:grid-cols-2"
      >
        <label className="flex flex-col gap-1 text-sm">
          Elevation gain — climb (ft)
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
          Elevation loss — descent (ft)
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
            step="1"
            value={tempF}
            onChange={(e) => setTempF(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          As-of date (optional)
          <input
            type="date"
            value={asOf}
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

      {result && (
        <section className="space-y-4">
          <div className="text-sm text-zinc-400">
            Predictions as of {result.as_of} · {result.temp_f}°F ({result.temp_source})
          </div>
          <div className="overflow-x-auto rounded-xl border border-zinc-800">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-zinc-900/80 text-zinc-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Distance</th>
                  <th className="px-4 py-3 font-medium">Predicted Time</th>
                  <th className="px-4 py-3 font-medium">Pace</th>
                </tr>
              </thead>
              <tbody>
                {result.predictions.map((row) => (
                  <tr key={row.distance_label} className="border-t border-zinc-800">
                    <td className="px-4 py-3">{row.distance_label}</td>
                    <td className="px-4 py-3">{formatTime(row.predicted_time_sec)}</td>
                    <td className="px-4 py-3">{formatPace(row.pace_min_per_mi)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
