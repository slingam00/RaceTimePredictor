"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ManualConditionsForm } from "@/components/ManualConditionsForm";
import { PredictionHorizonNotice } from "@/components/PredictionHorizonNotice";
import { PredictionsTable } from "@/components/PredictionsTable";
import {
  getPredictionHorizon,
  getRace,
  predictRaces,
  type PredictResponse,
  type RaceDetail,
} from "@/lib/api";
import { formatLocation, formatRaceDate, formatTempF } from "@/lib/format";

export default function RaceDetailPage() {
  const params = useParams<{ id: string }>();
  const raceId = Number(params.id);

  const [race, setRace] = useState<RaceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
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
    if (!Number.isFinite(raceId)) {
      setError("Invalid race id");
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    getRace(raceId)
      .then((detail) => {
        if (!cancelled) {
          setRace(detail);
          setShowManual(
            detail.elev_gain_ft == null ||
              detail.elev_loss_ft == null ||
              detail.warnings.length > 0
          );
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setRace(null);
          setError(err instanceof Error ? err.message : "Failed to load race");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [raceId]);

  async function onPredict() {
    if (!race) return;
    setPredicting(true);
    setError(null);
    try {
      const response = await predictRaces({ race_id: race.race_id });
      setResult(response);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Prediction failed");
      if (
        err instanceof Error &&
        err.message.toLowerCase().includes("elevation")
      ) {
        setShowManual(true);
      }
    } finally {
      setPredicting(false);
    }
  }

  if (loading) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-12 text-zinc-400">Loading race…</main>
    );
  }

  if (!race) {
    return (
      <main className="mx-auto max-w-4xl space-y-4 px-6 py-12">
        <Link href="/" className="text-sm text-zinc-400 hover:text-zinc-200">
          ← Back to search
        </Link>
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-red-200">
          {error ?? "Race not found"}
        </div>
      </main>
    );
  }

  const elevationKnown =
    race.elev_gain_ft != null && race.elev_loss_ft != null;
  const outsideHorizon =
    maxPredictionDate != null && race.race_date > maxPredictionDate;

  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col gap-8 px-6 py-12">
      <div>
        <Link href="/" className="text-sm text-zinc-400 hover:text-zinc-200">
          ← Back to search
        </Link>
      </div>

      <header className="space-y-3">
        <h1 className="text-3xl font-semibold">{race.name}</h1>
        <p className="text-zinc-400">
          {formatLocation(race.city, race.state)} · Race day {formatRaceDate(race.race_date)}
        </p>
        <div className="flex flex-wrap gap-2 text-sm">
          {elevationKnown ? (
            <span className="rounded-full bg-zinc-800 px-3 py-1 text-zinc-300">
              Elev {race.elev_gain_ft}↑ / {race.elev_loss_ft}↓ ft ({race.elev_source})
            </span>
          ) : (
            <span className="rounded-full bg-amber-950 px-3 py-1 text-amber-100">
              Elevation unknown
            </span>
          )}
          {race.temp_f != null && (
            <span className="rounded-full bg-zinc-800 px-3 py-1 text-zinc-300">
              {formatTempF(race.temp_f)}°F ({race.weather_source})
            </span>
          )}
        </div>
        {race.offered_events.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {race.offered_events.map((event) => (
              <span
                key={event.event_id}
                className="rounded-full border border-zinc-700 px-2.5 py-0.5 text-xs text-zinc-300"
              >
                {event.distance_label}
              </span>
            ))}
          </div>
        )}
      </header>

      <PredictionHorizonNotice message={horizonMessage} />

      {race.warnings.length > 0 && (
        <ul className="space-y-1 rounded-lg border border-amber-900/60 bg-amber-950/30 px-4 py-3 text-sm text-amber-100">
          {race.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      {elevationKnown && !outsideHorizon && (
        <div>
          <button
            type="button"
            onClick={onPredict}
            disabled={predicting}
            className="rounded-lg bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-500 disabled:opacity-60"
          >
            {predicting ? "Predicting…" : "Predict all distances"}
          </button>
        </div>
      )}

      {outsideHorizon && (
        <p className="text-sm text-zinc-400">
          This race is beyond your training lookback window and cannot be predicted
          automatically.
        </p>
      )}

      {error && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-red-200">
          {error}
        </div>
      )}

      {result && <PredictionsTable result={result} />}

      <section className="border-t border-zinc-800 pt-8">
        <button
          type="button"
          onClick={() => setShowManual((value) => !value)}
          className="text-sm text-zinc-400 underline-offset-4 hover:text-zinc-200 hover:underline"
        >
          {showManual ? "Hide manual override" : "Override course conditions"}
        </button>
        {showManual && (
          <div className="mt-6">
            <ManualConditionsForm
              raceId={race.race_id}
              defaultElevGainFt={race.elev_gain_ft}
              defaultElevLossFt={race.elev_loss_ft}
              defaultTempF={race.temp_f}
              defaultAsOf={race.race_date}
              maxAsOf={maxPredictionDate ?? undefined}
              title="Override course conditions"
            />
          </div>
        )}
      </section>
    </main>
  );
}
