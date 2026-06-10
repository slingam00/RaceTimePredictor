"use client";

import { formatRaceDate } from "@/lib/format";

type PredictionHorizonNoticeProps = {
  message?: string | null;
  maxPredictionDate?: string | null;
};

export function PredictionHorizonNotice({
  message,
  maxPredictionDate,
}: PredictionHorizonNoticeProps) {
  const text =
    message ??
    (maxPredictionDate
      ? `Predictions can only be made up until ${formatRaceDate(maxPredictionDate)}.`
      : null);

  if (!text) {
    return null;
  }

  return (
    <p className="rounded-lg border border-amber-900/60 bg-amber-950/30 px-4 py-3 text-sm text-amber-100">
      {text}
    </p>
  );
}
