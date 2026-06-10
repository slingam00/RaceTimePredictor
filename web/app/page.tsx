"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { searchRaces, getPredictionHorizon, type RaceSummary } from "@/lib/api";
import { formatLocation, formatRaceDate } from "@/lib/format";
import { ManualConditionsForm } from "@/components/ManualConditionsForm";
import { PredictionHorizonNotice } from "@/components/PredictionHorizonNotice";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [city, setCity] = useState("");
  const [state, setState] = useState("");
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showManual, setShowManual] = useState(false);
  const [horizonMessage, setHorizonMessage] = useState<string | null>(null);
  const [maxPredictionDate, setMaxPredictionDate] = useState<string | null>(null);

  useEffect(() => {
    getPredictionHorizon()
      .then((horizon) => {
        setHorizonMessage(horizon.prediction_horizon_message ?? null);
        setMaxPredictionDate(horizon.max_prediction_date ?? null);
      })
      .catch(() => {
        setHorizonMessage(null);
        setMaxPredictionDate(null);
      });
  }, []);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setRaces([]);
      setError(null);
      return;
    }

    const handle = window.setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await searchRaces({
          q: trimmed,
          city: city.trim() || undefined,
          state: state.trim() || undefined,
          start_date: new Date().toISOString().slice(0, 10),
          results_per_page: 20,
        });
        setRaces(response.races);
        if (response.prediction_horizon_message) {
          setHorizonMessage(response.prediction_horizon_message);
        }
      } catch (err) {
        setRaces([]);
        setError(err instanceof Error ? err.message : "Search failed");
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => window.clearTimeout(handle);
  }, [query, city, state]);

  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col gap-8 px-6 py-12">
      <header className="space-y-2">
        <p className="text-sm uppercase tracking-wide text-zinc-400">Race Time Predictor</p>
        <h1 className="text-3xl font-semibold">Find your upcoming race</h1>
        <p className="text-zinc-400">
          Search RunSignup races, review course conditions, and get predictions for 5K through
          marathon. API:{" "}
          <code className="rounded bg-zinc-900 px-1.5 py-0.5 text-zinc-200">{API_BASE}</code>
        </p>
      </header>

      <PredictionHorizonNotice message={horizonMessage} />

      <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
        <label className="flex flex-col gap-1 text-sm">
          Search races
          <input
            type="search"
            placeholder="e.g. Florida marathon, Boston 5K"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
            autoFocus
          />
        </label>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="flex flex-col gap-1 text-sm">
            City (optional)
            <input
              type="text"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            State (optional)
            <input
              type="text"
              value={state}
              onChange={(e) => setState(e.target.value)}
              className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2"
            />
          </label>
        </div>

        {query.trim().length < 2 && (
          <p className="text-sm text-zinc-500">Type at least 2 characters to search.</p>
        )}
        {loading && <p className="text-sm text-zinc-400">Searching…</p>}
        {error && (
          <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-red-200">
            {error}
          </div>
        )}
      </section>

      {races.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-medium">Upcoming races</h2>
          <ul className="divide-y divide-zinc-800 rounded-xl border border-zinc-800">
            {races.map((race) => (
              <li key={race.race_id}>
                <Link
                  href={`/races/${race.race_id}`}
                  className="block px-4 py-4 transition hover:bg-zinc-900/60"
                >
                  <div className="font-medium text-zinc-100">{race.name}</div>
                  <div className="mt-1 text-sm text-zinc-400">
                    {formatLocation(race.city, race.state)} · {formatRaceDate(race.next_date)}
                  </div>
                  {race.offered_distances.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {race.offered_distances.map((label) => (
                        <span
                          key={label}
                          className="rounded-full bg-zinc-800 px-2.5 py-0.5 text-xs text-zinc-300"
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="border-t border-zinc-800 pt-8">
        <button
          type="button"
          onClick={() => setShowManual((value) => !value)}
          className="text-sm text-zinc-400 underline-offset-4 hover:text-zinc-200 hover:underline"
        >
          {showManual ? "Hide manual entry" : "Enter course conditions manually"}
        </button>
        {showManual && (
          <div className="mt-6">
            <ManualConditionsForm maxAsOf={maxPredictionDate ?? undefined} />
          </div>
        )}
      </section>
    </main>
  );
}
